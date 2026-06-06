"""Helper functions for the enhanced contrastive dual-teacher main pipeline."""

import time
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

from llm_teacher import call_real_llm_api, create_llm_prompt_for_node

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_FILE_MAP = {
    "cora": "cora.pt",
    "citeseer": "citeseer.pt",
    "pubmed": "pubmed.pt",
    "wikics": "wikics.pt",
    "arxiv": "arxiv.pt",
}

def resolve_dataset_path(dataset_name: str) -> Tuple[str, str]:
    canonical_name = dataset_name.lower()
    if canonical_name not in DATASET_FILE_MAP:
        supported = sorted(list(DATASET_FILE_MAP.keys()))
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {supported}")
    return canonical_name, os.path.join(REPO_ROOT, "datasets", DATASET_FILE_MAP[canonical_name])


def safe_torch_load(path: str, map_location: Optional[str] = None):
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    try:
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)
    except RuntimeError as e:
        # Handle checkpoints saved from GPU environments when CUDA is unavailable.
        if "Attempting to deserialize object on a CUDA device" in str(e):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")
        raise


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_llm_confidence_stats(
    llm_predictions: torch.Tensor,
    llm_confidences: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    confidence_threshold: Optional[float] = None,
    prefix: str = "  ",
) -> None:
    """Print confidence distribution stats for current run logs."""
    valid_mask = (llm_predictions >= 0)
    if query_mask is not None:
        valid_mask = valid_mask & query_mask.bool()

    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        scope = "query nodes" if query_mask is not None else "all nodes"
        print(f"{prefix}LLM Confidence Stats ({scope}): N/A (no valid predictions)")
        return

    conf = llm_confidences[valid_mask].detach().float()
    mean_v = float(conf.mean().item())
    std_v = float(conf.std(unbiased=False).item())
    min_v = float(conf.min().item())
    max_v = float(conf.max().item())
    q25, q50, q75 = torch.quantile(conf, torch.tensor([0.25, 0.5, 0.75], device=conf.device))
    q10, q90 = torch.quantile(conf, torch.tensor([0.10, 0.90], device=conf.device))

    scope = "query nodes" if query_mask is not None else "all nodes"
    print(f"{prefix}LLM Confidence Stats ({scope}):")
    print(f"{prefix}  Count: {valid_count}")
    print(f"{prefix}  Mean±Std: {mean_v:.3f} ± {std_v:.3f}")
    print(f"{prefix}  Min/Q10/Q25/Median/Q75/Q90/Max: "
          f"{min_v:.3f}/{q10.item():.3f}/{q25.item():.3f}/{q50.item():.3f}/{q75.item():.3f}/{q90.item():.3f}/{max_v:.3f}")

    # Bucketed distribution
    b1 = int(((conf >= 0.0) & (conf < 0.3)).sum().item())
    b2 = int(((conf >= 0.3) & (conf < 0.5)).sum().item())
    b3 = int(((conf >= 0.5) & (conf < 0.7)).sum().item())
    b4 = int(((conf >= 0.7) & (conf < 0.9)).sum().item())
    b5 = int((conf >= 0.9).sum().item())
    print(f"{prefix}  Histogram [0.0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0]: "
          f"{b1}, {b2}, {b3}, {b4}, {b5}")

    # Pass counts for common thresholds + current run threshold
    threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    if confidence_threshold is not None:
        t = float(confidence_threshold)
        if all(abs(t - x) > 1e-9 for x in threshold_list):
            threshold_list.append(t)
    threshold_list = sorted(set(threshold_list))

    pass_parts = []
    for t in threshold_list:
        pass_n = int((conf >= t).sum().item())
        pass_parts.append(f">={t:.2f}:{pass_n}/{valid_count}")
    print(f"{prefix}  Pass@threshold: " + " | ".join(pass_parts))


def _masked_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[float, int]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    mask = mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return float("nan"), 0
    acc = float((predictions[mask] == labels[mask]).float().mean().item())
    return acc, count


def print_gnn_teacher_quality_stats(
    gnn_predictions: torch.Tensor,
    gnn_confidences: torch.Tensor,
    labels: torch.Tensor,
    train_pool_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    prefix: str = "  ",
) -> None:
    train_non_gt_mask = train_pool_mask.bool() & ~ground_truth_mask.bool()
    scopes = [
        ("train_pool_non_gt", train_non_gt_mask),
        ("train_pool_all", train_pool_mask.bool()),
        ("valid", val_mask.bool()),
        ("test", test_mask.bool()),
    ]

    print(f"{prefix}GNN Teacher Pseudo-label Stats:")
    for name, mask in scopes:
        acc, count = _masked_accuracy(gnn_predictions, labels, mask)
        if count == 0:
            print(f"{prefix}  {name}: N/A (n=0)")
        else:
            print(f"{prefix}  {name}: acc={acc:.3f} (n={count})")

    conf = gnn_confidences[train_non_gt_mask].detach().float()
    if conf.numel() == 0:
        print(f"{prefix}  train_pool_non_gt confidence: N/A (n=0)")
        return

    mean_v = float(conf.mean().item())
    std_v = float(conf.std(unbiased=False).item())
    min_v = float(conf.min().item())
    max_v = float(conf.max().item())
    q25, q50, q75 = torch.quantile(conf, torch.tensor([0.25, 0.5, 0.75], device=conf.device))
    q10, q90 = torch.quantile(conf, torch.tensor([0.10, 0.90], device=conf.device))
    print(
        f"{prefix}  train_pool_non_gt conf mean±std: {mean_v:.3f} ± {std_v:.3f}"
    )
    print(
        f"{prefix}  train_pool_non_gt conf min/Q10/Q25/Median/Q75/Q90/Max: "
        f"{min_v:.3f}/{q10.item():.3f}/{q25.item():.3f}/{q50.item():.3f}/"
        f"{q75.item():.3f}/{q90.item():.3f}/{max_v:.3f}"
    )

    threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    parts = []
    for t in threshold_list:
        mask_t = train_non_gt_mask & (gnn_confidences >= t)
        acc_t, count_t = _masked_accuracy(gnn_predictions, labels, mask_t)
        if count_t == 0:
            parts.append(f">={t:.2f}:0")
        else:
            parts.append(f">={t:.2f}:{count_t} (acc={acc_t:.3f})")
    print(f"{prefix}  train_pool_non_gt Pass@threshold: " + " | ".join(parts))


def save_gnn_teacher_artifact(
    artifact_path: str,
    args,
    data,
    gnn_predictions: torch.Tensor,
    gnn_confidences: torch.Tensor,
    gnn_class_probabilities: Optional[torch.Tensor] = None,
) -> None:
    artifact = {
        "dataset": args.dataset,
        "ablation_mode": args.ablation_mode,
        "seed": args.seed,
        "k_shot": args.k_shot,
        "query_ratio_requested": args.query_ratio,
        "query_count": int(data.query_mask.sum().item()),
        "confidence_threshold": args.confidence_threshold,
        "use_gnn_fallback_labels": args.use_gnn_fallback_labels,
        "gnn_fallback_top_percentile": args.gnn_fallback_top_percentile,
        "gnn_fallback_weight": args.gnn_fallback_weight,
        "struct_node2vec_dim": args.struct_node2vec_dim,
        "gnn_predictions": gnn_predictions.detach().cpu(),
        "gnn_confidences": gnn_confidences.detach().cpu(),
        "gnn_class_probabilities": None if gnn_class_probabilities is None else gnn_class_probabilities.detach().cpu(),
        "labels": data.y.detach().cpu(),
        "train_mask": data.train_mask.detach().cpu(),
        "train_pool_mask": data.train_pool_mask.detach().cpu(),
        "val_mask": data.val_mask.detach().cpu(),
        "test_mask": data.test_mask.detach().cpu(),
        "query_mask": data.query_mask.detach().cpu(),
        "supervision_mask": data.supervision_mask.detach().cpu(),
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    torch.save(artifact, artifact_path)


def load_llm_predictions(dataset_name: str, data: Any, query_ratio: float = 0.1,
                        prompt_type: str = "zero_shot_top3_v1",
                        cache_id: int = None, cache_read_only: bool = False,
                        llm_logger: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from llm_prompts import LLM_PROMPT_STRATEGY
    prompt_type = LLM_PROMPT_STRATEGY

    from llm_cache_manager import LLMCacheManager
    cache_manager = LLMCacheManager()

    # Prefer dataset-provided label order to keep label index mapping consistent.
    from llm_prompts import get_dataset_metadata
    # dataset_name is already the parameter
    try:
        metadata = get_dataset_metadata(dataset_name)
        label_names = metadata['label_names']
    except Exception:
        # Fallback if metadata not found
        label_names = []
    if hasattr(data, "label_name") and isinstance(data.label_name, (list, tuple)) and len(data.label_name) > 0:
        label_names = list(data.label_name)
    elif "label_name" in data and isinstance(data["label_name"], (list, tuple)) and len(data["label_name"]) > 0:
        label_names = list(data["label_name"])
    elif hasattr(data, "label_names") and isinstance(data.label_names, (list, tuple)) and len(data.label_names) > 0:
        label_names = list(data.label_names)
    elif hasattr(data, "category_names") and isinstance(data.category_names, (list, tuple)) and len(data.category_names) > 0:
        label_names = list(data.category_names)
    
    expected_query_indices = torch.where(data.query_mask.to(data.x.device))[0]
    actual_query_k = int(expected_query_indices.numel())
    
    # SBERT dimension for reasoning embeddings
    from llm_teacher.sbert_embedder import get_embedder
    embedder = get_embedder(device=data.x.device)
    embedding_dim = embedder.embedding_dim

    if actual_query_k == 0:
        predictions = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
        confidences = torch.zeros(data.num_nodes, device=data.x.device)
        class_probabilities = torch.zeros((data.num_nodes, len(label_names)), dtype=torch.float32, device=data.x.device)
        reasoning_embeddings = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32, device=data.x.device)
        print("  ✓ Query count is 0. Skipping LLM API/cache and returning empty pseudo-labels.")
        return predictions, confidences, class_probabilities, reasoning_embeddings

    if cache_id is not None:
        cache_path = cache_manager.get_cache_path(
            dataset_name,
            prompt_type,
            query_ratio=query_ratio,
            query_count=actual_query_k,
            cache_id=cache_id,
        )
        if os.path.exists(cache_path):
            cache = safe_torch_load(cache_path)
        else:
            cache = None
    else:
        cache = cache_manager.load_cache(
            dataset_name,
            prompt_type,
            query_ratio=query_ratio,
            query_count=actual_query_k,
        )

    num_classes = len(label_names)
    predictions = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
    confidences = torch.zeros(data.num_nodes, device=data.x.device)
    class_probabilities = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32, device=data.x.device)
    reasoning_embeddings = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32, device=data.x.device)

    reused_indices_list = []
    
    if cache:
        if isinstance(cache, dict) and 'predictions' in cache:
            cached_predictions_full = cache['predictions'].to(data.x.device)
            cached_query_count = cache.get('query_count')
            cached_node_indices = cache.get('node_indices', [])
            
            if cached_query_count == actual_query_k:
                print(f"  ✓ Cache query count matches ({actual_query_k}). Forcing 100% reuse via sequential mapping.")
                
                target_device = data.x.device
                if cached_predictions_full.size(0) == data.num_nodes:
                    valid_mask = (cached_predictions_full >= 0)
                    valid_indices = torch.where(valid_mask)[0]
                    valid_preds = cached_predictions_full[valid_mask]
                    
                    # Ensure indexed tensors are on the same device as indices (target_device)
                    valid_confs = cache['confidences'].to(target_device)[valid_mask] if 'confidences' in cache else torch.ones_like(valid_preds, dtype=torch.float32)
                    valid_probs = cache['class_probabilities'].to(target_device)[valid_mask] if 'class_probabilities' in cache else None
                    valid_embs = cache['reasoning_embeddings'].to(target_device)[valid_mask] if 'reasoning_embeddings' in cache else None
                else:
                    valid_preds = cached_predictions_full
                    valid_indices = torch.tensor(cached_node_indices, dtype=torch.long, device=target_device) if cached_node_indices else torch.arange(valid_preds.size(0), device=target_device)
                    valid_confs = cache['confidences'].to(target_device) if 'confidences' in cache else torch.ones_like(valid_preds, dtype=torch.float32)
                    valid_probs = cache['class_probabilities'].to(target_device) if 'class_probabilities' in cache else None
                    valid_embs = cache['reasoning_embeddings'].to(target_device) if 'reasoning_embeddings' in cache else None

                num_to_reuse = min(valid_preds.size(0), actual_query_k)
                reused_node_indices = []
                for i in range(num_to_reuse):
                    idx_val = int(valid_indices[i].item())
                    predictions[idx_val] = valid_preds[i]
                    confidences[idx_val] = valid_confs[i]
                    if valid_probs is not None: class_probabilities[idx_val] = valid_probs[i].to(data.x.device)
                    if valid_embs is not None: reasoning_embeddings[idx_val] = valid_embs[i].to(data.x.device)
                    reused_node_indices.append(idx_val)
                
                if len(reused_node_indices) == actual_query_k:
                    print(f"  ✓ Successfully forced 100% reuse of {actual_query_k} nodes.")
                    
                    # [CRITICAL] Update the query_mask IN-PLACE to ensure all references (like in main()) are updated
                    # This ensures we use the EXACT nodes that were already cached, saving LLM costs.
                    data.query_mask.fill_(False)
                    reused_tensor = torch.tensor(reused_node_indices, dtype=torch.long, device=data.query_mask.device)
                    data.query_mask[reused_tensor] = True
                    
                    # Also update unqueried_mask to maintain consistency
                    if hasattr(data, 'unqueried_mask') and hasattr(data, 'train_pool_mask'):
                        new_unqueried = data.train_pool_mask.to(data.query_mask.device) & ~data.query_mask
                        data.unqueried_mask.copy_(new_unqueried)
                        
                    return predictions, confidences, class_probabilities, reasoning_embeddings
                else:
                    print(f"  ⚠ Forced reuse only covered {len(reused_node_indices)}/{actual_query_k} nodes.")

        if len(reused_indices_list) < actual_query_k and isinstance(cache, dict) and any(isinstance(k, str) and k.isdigit() for k in cache.keys()):
            print(f"  Using Global Node Pool for individual node lookup...")
            for idx in expected_query_indices:
                idx_val = int(idx.item())
                idx_str = str(idx_val)
                if idx_str in cache:
                    node_data = cache[idx_str]
                    predictions[idx_val] = node_data['prediction']
                    confidences[idx_val] = node_data['confidence']
                    if 'class_probabilities' in node_data and node_data['class_probabilities'] is not None:
                        cp = node_data['class_probabilities']
                        if cp.shape[0] == num_classes:
                            class_probabilities[idx_val] = cp.to(data.x.device)
                    if 'reasoning_embeddings' in node_data and node_data['reasoning_embeddings'] is not None:
                        re = node_data['reasoning_embeddings']
                        if re.shape[0] == embedding_dim:
                            reasoning_embeddings[idx_val] = re.to(data.x.device)
                    reused_indices_list.append(idx_val)
        
        elif isinstance(cache, dict) and 'predictions' in cache:
            try:
                cached_label_names = cache.get('label_names', None)
                if cached_label_names is not None and list(cached_label_names) == list(label_names):
                    cached_predictions_full = cache['predictions'].to(data.x.device)
                    cached_node_indices = cache.get('node_indices', [])
                    
                    if len(cached_node_indices) == actual_query_k:
                        print(f"  Set-based cache count matches ({actual_query_k}). Reusing mapping.")
                        for i, target_idx in enumerate(expected_query_indices):
                            source_idx = i # cache['predictions'] usually maps 1:1 if it was saved as a subset
                            # Note: cache may store full num_nodes tensor or subset. 
                            # If it's a subset, we need to know the mapping.
                            if source_idx < cached_predictions_full.size(0):
                                target_idx_val = int(target_idx.item())
                                predictions[target_idx_val] = cached_predictions_full[source_idx]
                                reused_indices_list.append(target_idx_val)
                    else:
                        for idx in expected_query_indices:
                            idx_val = int(idx.item())
                            if idx_val < cached_predictions_full.size(0) and cached_predictions_full[idx_val] >= 0:
                                predictions[idx_val] = cached_predictions_full[idx_val]
                                reused_indices_list.append(idx_val)
            except Exception as e:
                print(f"  Failed to load set-based cache: {e}")

    reused_indices = torch.tensor(reused_indices_list, dtype=torch.long, device=data.x.device)
    if reused_indices.numel() > 0:
        reused_indices = torch.unique(reused_indices)
    
    # Calculate missing_indices: expected_query_indices NOT in reused_indices
    reused_set = set(reused_indices.tolist())
    missing_indices_list = [idx.item() for idx in expected_query_indices if int(idx.item()) not in reused_set]
    missing_indices = torch.tensor(missing_indices_list, dtype=torch.long, device=data.x.device)

    print(f"  Cache results: Reused {reused_indices.numel()} nodes, Missing {missing_indices.numel()} nodes.")

    if cache_read_only and missing_indices.numel() > 0:
        print(f"  cache_read_only=True: {missing_indices.numel()} nodes are missing and will remain unlabeled.")
        return predictions, confidences, class_probabilities, reasoning_embeddings
    
    # If not cache_read_only, we proceed to query only the missing nodes
    query_indices = missing_indices
    query_count = query_indices.numel()
    
    if query_count == 0:
        valid_count = (predictions >= 0).sum().item()
        print("  ✓ All required query nodes are served from cache.")
        print(f"  Valid LLM predictions: {valid_count}/{data.num_nodes}")
        return predictions, confidences, class_probabilities, reasoning_embeddings
    
    print(f"🚀 Starting LLM API calls for {query_count} query nodes...", flush=True)
    if query_ratio is not None:
        print(f"📝 Dataset: {dataset_name.upper()} | Query Count: {actual_query_k} | Query Ratio(ref): {query_ratio}", flush=True)
    else:
        print(f"📝 Dataset: {dataset_name.upper()} | Query Count: {actual_query_k}", flush=True)
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}", flush=True)
    
    from llm_prompts import parse_llm_response
    
    llm_inputs = []
    llm_outputs = []
    queried_node_indices = []
    new_reasoning_embs = []
    parse_fail_count = 0

    def _clip_confidence(raw_conf: float) -> float:
        return float(max(0.0, min(1.0, raw_conf)))

    def _fallback_distribution(top_idx: int, top_conf: float, n_cls: int):
        if n_cls <= 0 or top_idx < 0 or top_idx >= n_cls:
            return None
        top_conf = float(max(0.0, min(1.0, top_conf)))
        if n_cls == 1:
            return [1.0]
        remain = max(0.0, 1.0 - top_conf)
        vec = np.full(n_cls, remain / (n_cls - 1), dtype=float)
        vec[top_idx] = top_conf
        s = float(vec.sum())
        if s <= 0:
            return None
        vec = vec / s
        return vec.tolist()
    
    k_shot_examples = []
    if hasattr(data, 'train_mask') and hasattr(data, 'raw_texts') and hasattr(data, 'y'):
        train_indices = torch.where(data.train_mask)[0]
        
        label_to_indices = {}
        for idx in train_indices:
            if idx < len(data.raw_texts):
                label = data.y[idx].item()
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
        
        examples_per_class = 1
        for label, indices in label_to_indices.items():
            if len(k_shot_examples) >= len(label_names):
                break
            if indices:
                import random
                random.seed(42)
                selected_idx = random.choice(indices)
                if selected_idx < len(data.raw_texts):
                    k_shot_examples.append({
                        'text': data.raw_texts[selected_idx],
                        'label': label_names[label]
                    })
        
        print(f"  Selected {len(k_shot_examples)} k-shot examples (1 per class)")
    
    def process_node(i, idx):
        if idx >= data.num_nodes:
            return i, idx, -1, 0.0, torch.zeros(embedding_dim), f"Invalid index {idx}"
        
        try:
            node_text = None
            if hasattr(data, 'raw_texts') and idx < len(data.raw_texts):
                node_text = data.raw_texts[idx]
            
            prompt = create_llm_prompt_for_node(
                dataset_name,
                label_names,
                node_text,
                k_shot_examples=k_shot_examples
            )
            
            try:
                response = call_real_llm_api(prompt)
                llm_output = response.get('content', '')
            except Exception as e:
                if "insufficient_quota" in str(e).lower() or "429" in str(e):
                    # Quota error is critical, return a specific signal
                    return i, idx, -1, 0.0, torch.zeros(embedding_dim), f"QUOTA_EXCEEDED: {str(e)}"
                raise e
            
            parsed = parse_llm_response(llm_output, label_names)
            pred_idx = parsed.get('answer', -1)
            if isinstance(pred_idx, str) and pred_idx in label_names:
                pred_idx = label_names.index(pred_idx)
            conf = _clip_confidence(parsed.get('confidence', 0.0))
            
            reasoning_text = parsed.get('reasoning', '')
            r_emb = torch.zeros(embedding_dim)
            if reasoning_text:
                r_emb = embedder.embed(reasoning_text).cpu()
            
            return i, idx, pred_idx, conf, r_emb, llm_output
        except Exception as e:
            return i, idx, -1, 0.0, torch.zeros(embedding_dim), str(e)

    max_workers = 10
    results_ordered = [None] * len(query_indices)
    quota_exceeded = False
    
    print(f"  Parallelizing LLM queries with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {executor.submit(process_node, i, idx): i for i, idx in enumerate(query_indices)}
        
        completed = 0
        for future in as_completed(future_to_node):
            try:
                res = future.result()
                i, idx, pred_idx, conf, r_emb, llm_output = res
                
                if isinstance(llm_output, str) and "QUOTA_EXCEEDED" in llm_output:
                    quota_exceeded = True
                    # If quota exceeded, we should probably stop submitting new tasks
                    # but for now we just collect the failure
                
                results_ordered[i] = (idx, pred_idx, conf, r_emb, llm_output)
            except Exception as e:
                print(f"  ⚠️ Unexpected error in thread: {e}")
            
            completed += 1
            if completed % 10 == 0 or completed == len(query_indices):
                progress_pct = 100 * completed / len(query_indices)
                print(f"🔄 LLM Query Progress: {completed}/{len(query_indices)} ({progress_pct:.1f}%)", flush=True)
            
            if quota_exceeded:
                print("  🛑 OpenAI Quota Exceeded. Stopping further queries and using available results.")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    for res in results_ordered:
        if res is None: continue
        idx, pred_idx, conf, r_emb, llm_output = res
        
        idx_val = int(idx.item())
        predictions[idx_val] = pred_idx
        confidences[idx_val] = conf
        reasoning_embeddings[idx_val] = r_emb.to(data.x.device)
        
        # class_probabilities (fallback distribution)
        dist = _fallback_distribution(pred_idx, conf, num_classes)
        if dist:
            class_probabilities[idx_val] = torch.tensor(dist, dtype=torch.float32, device=data.x.device)
        
        llm_inputs.append(f"Node {idx_val} (Parallel processing)")
        llm_outputs.append(llm_output)
        queried_node_indices.append(idx_val)
        if pred_idx == -1:
            parse_fail_count += 1

    queried_success = int((predictions[query_indices] >= 0).sum().item())
    total_count = len(query_indices)
    success_rate = (queried_success / total_count * 100) if total_count > 0 else 0
    
    print(f"\n✅ LLM API calls completed!", flush=True)
    print(f"📊 Results: {queried_success}/{total_count} successful ({success_rate:.1f}%)", flush=True)
    if parse_fail_count > 0:
        print(f"⚠️ Parse failures: {parse_fail_count}/{total_count}", flush=True)
    print(f"⏰ Completed at: {time.strftime('%H:%M:%S')}", flush=True)
    
    cached_predictions_full = None
    cached_class_probabilities_full = None
    cached_reasoning_embeddings_full = None
    
    if cache:
        if isinstance(cache, dict) and any(isinstance(k, str) and k.isdigit() for k in cache.keys()):
            cached_predictions_full = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
            cached_class_probabilities_full = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32, device=data.x.device)
            cached_reasoning_embeddings_full = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32, device=data.x.device)
            
            for idx_str, node_data in cache.items():
                try:
                    idx_val = int(idx_str)
                    if idx_val < data.num_nodes:
                        cached_predictions_full[idx_val] = node_data.get('prediction', -1)
                        if 'class_probabilities' in node_data and node_data['class_probabilities'] is not None:
                            cp = node_data['class_probabilities']
                            if cp.shape[0] == num_classes:
                                cached_class_probabilities_full[idx_val] = cp.to(data.x.device)
                        if 'reasoning_embeddings' in node_data and node_data['reasoning_embeddings'] is not None:
                            re = node_data['reasoning_embeddings']
                            if re.shape[0] == embedding_dim:
                                cached_reasoning_embeddings_full[idx_val] = re.to(data.x.device)
                except (ValueError, KeyError, AttributeError):
                    continue
        
        elif isinstance(cache, dict) and 'predictions' in cache:
            try:
                cached_label_names = cache.get('label_names', None)
                if cached_label_names is not None and list(cached_label_names) == list(label_names):
                    cached_predictions_full = cache['predictions'].to(data.x.device)
                    cached_class_probabilities_full = cache.get('class_probabilities', None)
                    if cached_class_probabilities_full is not None:
                        cached_class_probabilities_full = cached_class_probabilities_full.to(data.x.device)
                    cached_reasoning_embeddings_full = cache.get('reasoning_embeddings', None)
                    if cached_reasoning_embeddings_full is not None:
                        cached_reasoning_embeddings_full = cached_reasoning_embeddings_full.to(data.x.device)
            except Exception as e:
                print(f"  Failed to load set-based cache: {e}")
    
    if cached_predictions_full is not None:
        predictions_to_save = cached_predictions_full.clone()
        if cached_class_probabilities_full is not None:
            class_probabilities_to_save = cached_class_probabilities_full.clone()
        else:
            class_probabilities_to_save = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32, device=data.x.device)
        
        if cached_reasoning_embeddings_full is not None:
            reasoning_embeddings_to_save = cached_reasoning_embeddings_full.clone()
        else:
            reasoning_embeddings_to_save = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32, device=data.x.device)
    else:
        predictions_to_save = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
        class_probabilities_to_save = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32, device=data.x.device)
        reasoning_embeddings_to_save = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32, device=data.x.device)

    new_valid_mask = predictions >= 0
    predictions_to_save[new_valid_mask] = predictions[new_valid_mask]
    
    if new_valid_mask.any():
        class_probabilities_to_save[new_valid_mask] = class_probabilities[new_valid_mask]
    
    reasoning_valid_mask = reasoning_embeddings.abs().sum(dim=1) > 0
    reasoning_embeddings_to_save[reasoning_valid_mask] = reasoning_embeddings[reasoning_valid_mask]

    cache_manager.save_cache(
        dataset_name=dataset_name,
        prompt_type=prompt_type,
        query_ratio=query_ratio,
        query_count=actual_query_k,
        predictions=predictions_to_save.cpu(),
        confidences=torch.zeros_like(predictions_to_save).float().cpu(), # Placeholder
        llm_inputs=llm_inputs,
        llm_outputs=llm_outputs,
        node_indices=queried_node_indices,
        query_indices_all=expected_query_indices.detach().cpu().tolist(),
        num_nodes=data.num_nodes,
        label_names=label_names,
        cache_id=cache_id,
        class_probabilities=class_probabilities_to_save.cpu(),
        reasoning_embeddings=reasoning_embeddings_to_save.cpu(),
        true_labels=data.y.cpu() if hasattr(data, 'y') else None
    )
    
    return predictions, torch.zeros_like(predictions).float(), class_probabilities, reasoning_embeddings

def _normalize_adj_for_aax(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    values = torch.ones(row.size(0), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (num_nodes, num_nodes),
    ).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    deg_inv_sqrt = degree.pow(-0.5)
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        torch.stack([row, col]),
        norm_values,
        (num_nodes, num_nodes),
    ).coalesce()


def _compute_locle_aax_features(
    data,
    feature_source: str = "structural",
) -> torch.Tensor:
    num_nodes = int(data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0))
    base_x = getattr(data, "structural_x", None) if feature_source == "structural" else None
    if base_x is None:
        base_x = data.x
    base_x = base_x.detach().cpu().float()
    norm_adj = _normalize_adj_for_aax(data.edge_index, num_nodes)
    ax = torch.sparse.mm(norm_adj, base_x)
    aax = torch.sparse.mm(norm_adj, ax)
    return F.normalize(aax, p=2, dim=-1)


def _select_query_nodes(
    data,
    query_candidates: torch.Tensor,
    query_size: int,
    seed: int,
    method: str = "random",
    feature_source: str = "structural",
    num_clusters: int = 0,
) -> torch.Tensor:
    if query_size <= 0 or query_candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    if method == "random":
        query_perm = torch.randperm(
            query_candidates.numel(),
            generator=torch.Generator().manual_seed(seed + 1),
        )
        return query_candidates[query_perm[:query_size]]

    if method == "cluster_random":
        if KMeans is None:
            raise RuntimeError(f"scikit-learn is required for query_selection_method={method}")

        base_x = getattr(data, "structural_x", None) if feature_source == "structural" else None
        if base_x is None:
            base_x = data.x
        features = F.normalize(base_x.detach().cpu().float(), p=2, dim=-1)
        candidate_features = features[query_candidates]

        if int(query_candidates.numel()) <= query_size:
            return query_candidates.clone()

        if num_clusters is None or int(num_clusters) <= 0:
            inferred = len(torch.unique(data.y.cpu())) if hasattr(data, "y") else 8
            num_clusters = max(1, min(int(inferred), int(query_candidates.numel())))
        else:
            num_clusters = max(1, min(int(num_clusters), int(query_candidates.numel())))

        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=seed, n_init=10)
        labels = kmeans.fit_predict(candidate_features.numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=candidate_features.dtype)
        labels_t = torch.tensor(labels, dtype=torch.long)

        cluster_local_indices = []
        cluster_center_choice = []
        cluster_sizes = []
        for cluster_idx in range(num_clusters):
            local_idx = torch.where(labels_t == cluster_idx)[0]
            cluster_local_indices.append(local_idx)
            cluster_sizes.append(int(local_idx.numel()))
            feats = candidate_features[local_idx]
            center = centers[cluster_idx].unsqueeze(0)
            dist = torch.norm(feats - center, dim=1)
            order = torch.argsort(dist)
            cluster_center_choice.append(local_idx[order[0]])

        selected_local = set(int(idx.item()) for idx in cluster_center_choice if idx.numel() != 0)

        if len(selected_local) >= query_size:
            ordered = list(selected_local)[:query_size]
            return query_candidates[torch.tensor(ordered, dtype=torch.long)]

        remaining_budget = query_size - len(selected_local)
        size_total = max(1, sum(cluster_sizes))
        quotas = [0 for _ in range(num_clusters)]
        for cluster_idx, size in enumerate(cluster_sizes):
            quotas[cluster_idx] = int(round(remaining_budget * (size / size_total)))

        quota_sum = sum(quotas)
        if quota_sum < remaining_budget:
            order = sorted(range(num_clusters), key=lambda i: cluster_sizes[i], reverse=True)
            ptr = 0
            while quota_sum < remaining_budget and order:
                quotas[order[ptr % len(order)]] += 1
                quota_sum += 1
                ptr += 1
        elif quota_sum > remaining_budget:
            order = sorted(range(num_clusters), key=lambda i: quotas[i], reverse=True)
            ptr = 0
            while quota_sum > remaining_budget and order:
                idx = order[ptr % len(order)]
                if quotas[idx] > 0:
                    quotas[idx] -= 1
                    quota_sum -= 1
                ptr += 1

        additional_ranked = []
        for cluster_idx in range(num_clusters):
            local_idx = cluster_local_indices[cluster_idx]
            if local_idx.numel() == 0:
                continue
            order = torch.randperm(
                local_idx.numel(),
                generator=torch.Generator().manual_seed(seed + 101 + cluster_idx),
            ).tolist()
            ranked = [int(local_idx[o].item()) for o in order if int(local_idx[o].item()) not in selected_local]
            take = min(quotas[cluster_idx], len(ranked))
            chosen = ranked[:take]
            selected_local.update(chosen)
            additional_ranked.extend(chosen)

        if len(selected_local) < query_size:
            global_center = candidate_features.mean(dim=0, keepdim=True)
            global_dist = torch.norm(candidate_features - global_center, dim=1)
            global_closeness = 1.0 / (1.0 + global_dist)
            order = torch.argsort(global_closeness, descending=True)
            for idx in order.tolist():
                if idx not in selected_local:
                    selected_local.add(int(idx))
                if len(selected_local) >= query_size:
                    break

        center_ordered = [int(idx.item()) for idx in cluster_center_choice if idx.numel() != 0]
        final_local = []
        seen = set()
        for idx in center_ordered + additional_ranked:
            if idx not in seen:
                final_local.append(idx)
                seen.add(idx)
        if len(final_local) < query_size:
            for idx in sorted(selected_local):
                if idx not in seen:
                    final_local.append(idx)
                    seen.add(idx)
                if len(final_local) >= query_size:
                    break
        return query_candidates[torch.tensor(final_local[:query_size], dtype=torch.long)]

    raise ValueError(f"Unsupported query_selection_method: {method}")


def create_few_shot_split(
    data,
    query_ratio: float = None,
    k_shot: int = 0,
    dataset_name: str = "cora",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    query_budget_base_size: int = 0,
    seed: int = 42,
    query_selection_method: str = "random",
    query_selection_feature_source: str = "structural",
    query_selection_num_clusters: int = 0,
    teacher_mode: str = "dual",
):
    """Node split with optional class-wise k-shot examples.


    """
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    if not hasattr(data, 'y'):
        raise ValueError("Data object must have labels `y` for evaluation.")
    labels = data.y.cpu()
    dataset_name = dataset_name.lower()

    if query_ratio is None:
        query_ratio = 0.0
    if query_ratio is not None and not (0.0 <= query_ratio <= 1.0):
        raise ValueError("query_ratio must be in [0,1].")
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("train_ratio/val_ratio/test_ratio must be >= 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    original_train_mask = getattr(data, "train_mask", None)
    original_val_mask   = getattr(data, "val_mask",   None)
    original_test_mask  = getattr(data, "test_mask",  None)
    has_official_split = (
        isinstance(original_train_mask, torch.Tensor)
        and isinstance(original_val_mask,   torch.Tensor)
        and isinstance(original_test_mask,  torch.Tensor)
        and original_train_mask.numel() == num_nodes
        and original_val_mask.numel()   == num_nodes
        and original_test_mask.numel()  == num_nodes
    )

    ogb_datasets = {"arxiv", "ogbn-arxiv", "products", "ogbn-products"}
    is_ogb = dataset_name in ogb_datasets

    ground_truth_mask = torch.zeros(num_nodes, dtype=torch.bool)
    k_shot_mask = torch.zeros(num_nodes, dtype=torch.bool)

    use_official_split = is_ogb
    if is_ogb and not has_official_split:
        print(
            f"[WARN] OGB dataset({dataset_name}) expected official split, "
            f"but not found. Falling back to fixed ratio split."
        )
        use_official_split = False

    if use_official_split:
        off_train = original_train_mask.cpu().bool()
        val_mask  = original_val_mask.cpu().bool().clone()
        test_mask = original_test_mask.cpu().bool().clone()

        train_pool_mask = ~(val_mask | test_mask)

    else:
        train_pool_target = int(num_nodes * train_ratio)
        val_target = int(num_nodes * val_ratio)
        test_target = num_nodes - train_pool_target - val_target
        if train_pool_target <= 0 or val_target <= 0 or test_target <= 0:
            raise ValueError(
                f"Invalid split sizes from ratios: train={train_pool_target}, "
                f"val={val_target}, test={test_target}. Check train/val/test ratios."
            )

        split_perm = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed + 17))
        val_indices = split_perm[:val_target]
        test_indices = split_perm[val_target:val_target + test_target]

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        train_pool_mask = ~(val_mask | test_mask)

    if k_shot is not None and int(k_shot) > 0:
        k = int(k_shot)
        rng = torch.Generator().manual_seed(seed + 13)
        train_indices = torch.where(train_pool_mask)[0]
        train_labels = labels[train_indices]
        unique_labels = torch.unique(train_labels).tolist()
        for cls in unique_labels:
            cls = int(cls)
            cls_indices = train_indices[train_labels == cls]
            if cls_indices.numel() == 0:
                continue
            take = min(k, int(cls_indices.numel()))
            perm = torch.randperm(int(cls_indices.numel()), generator=rng)
            selected = cls_indices[perm[:take]]
            k_shot_mask[selected] = True
        ground_truth_mask = k_shot_mask.clone()

    if teacher_mode == "llm_only":
        all_query_candidates = torch.where(train_pool_mask)[0]
    else:
        all_query_candidates = torch.where(train_pool_mask & ~k_shot_mask)[0]

    if query_budget_base_size is not None and query_budget_base_size > 0:
        candidate_pool_size = min(int(query_budget_base_size), all_query_candidates.numel())
        if candidate_pool_size < all_query_candidates.numel():
            base_perm = torch.randperm(
                all_query_candidates.numel(),
                generator=torch.Generator().manual_seed(seed + 23)
            )
            query_candidates = all_query_candidates[base_perm[:candidate_pool_size]]
        else:
            query_candidates = all_query_candidates
    else:
        query_candidates = all_query_candidates

    budget_base_size_effective = (
        int(query_budget_base_size)
        if (query_budget_base_size is not None and query_budget_base_size > 0)
        else int(all_query_candidates.numel())
    )

    ratio = float(query_ratio) if query_ratio is not None else 0.0
    query_size = int(budget_base_size_effective * ratio)
    if ratio > 0 and query_size == 0 and budget_base_size_effective > 0:
        query_size = 1
    query_size = min(query_size, query_candidates.numel())

    query_indices = _select_query_nodes(
        data=data,
        query_candidates=query_candidates,
        query_size=query_size,
        seed=seed,
        method=query_selection_method,
        feature_source=query_selection_feature_source,
        num_clusters=query_selection_num_clusters,
    )

    query_mask = torch.zeros(num_nodes, dtype=torch.bool)
    query_mask[query_indices]     = True
    supervision_mask = query_mask.clone()

    for attr_name in [
        'query_mask', 'test_mask', 'train_mask', 'val_mask',
        'test_masks', 'train_masks', 'val_masks',
        'supervision_mask', 'ground_truth_mask', 'train_pool_mask',
        'query_candidate_mask', 'all_query_candidate_mask', 'k_shot_mask'
    ]:
        if hasattr(data, attr_name):
            delattr(data, attr_name)

    data.query_mask        = query_mask
    data.unqueried_mask    = train_pool_mask & ~query_mask
    data.train_mask        = train_pool_mask  # Following user's 60% split definition
    data.val_mask          = val_mask
    data.test_mask         = test_mask
    data.k_shot_mask       = k_shot_mask
    data.supervision_mask  = supervision_mask
    data.ground_truth_mask = ground_truth_mask
    data.train_pool_mask   = train_pool_mask
    data.query_candidate_mask     = torch.zeros(num_nodes, dtype=torch.bool)
    data.query_candidate_mask[query_candidates] = True
    data.all_query_candidate_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.all_query_candidate_mask[all_query_candidates] = True
    data.query_budget_base_size_effective = int(budget_base_size_effective)
    data.query_selection_method = str(query_selection_method)
    data.query_selection_feature_source = str(query_selection_feature_source)
    data.query_selection_num_clusters = int(query_selection_num_clusters)

    return data
