"""Helper functions for the enhanced contrastive dual-teacher main pipeline."""

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from typing import List, Dict, Any, Tuple, Optional
try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

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
