"""Dataset loading and split utilities for the dual-teacher SSL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import List, Tuple

import numpy as np

import networkx as nx
import torch
from torch_geometric.utils import to_networkx, to_undirected

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
except Exception:  # pragma: no cover
    sp = None
    eigsh = None

from llm_prompts import get_dataset_metadata
from llm_teacher.sbert_embedder import get_embedder
from pipeline.main_helpers import create_few_shot_split, resolve_dataset_path, safe_torch_load


@dataclass
class NodeTextRecord:
    """Structured title/abstract view for a TAG node."""

    title: str
    abstract: str

    @property
    def combined(self) -> str:
        return f"Title: {self.title}\nAbstract: {self.abstract}"


def _to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return list(value)


def resolve_label_names(data, dataset_name: str) -> List[str]:
    """Resolve canonical label names from the dataset object or metadata."""
    for attr in ("label_names", "label_name", "category_names"):
        if hasattr(data, attr):
            values = getattr(data, attr)
            if isinstance(values, (list, tuple)) and len(values) > 0:
                return [str(v) for v in values]

    metadata = get_dataset_metadata(dataset_name)
    return [str(v) for v in metadata["label_names"]]


def split_raw_text(raw_text: str) -> NodeTextRecord:
    """Parse a raw TAG text blob into title/abstract fields."""
    text = (raw_text or "").strip()
    if not text:
        return NodeTextRecord(title="Untitled", abstract="No abstract available.")

    for separator in (" : ", ":\n", ": "):
        if separator in text:
            title, abstract = text.split(separator, 1)
            title = title.strip() or "Untitled"
            abstract = abstract.strip() or "No abstract available."
            return NodeTextRecord(title=title, abstract=abstract)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return NodeTextRecord(title=lines[0], abstract=" ".join(lines[1:]))
    return NodeTextRecord(title=text[:160], abstract=text)


def ensure_text_fields(data) -> None:
    """Populate structured title/abstract arrays on the PyG data object."""
    raw_texts = _to_list(getattr(data, "raw_texts", []))
    records = [split_raw_text(text) for text in raw_texts]

    data.raw_texts = [record.combined for record in records]
    data.title_texts = [record.title for record in records]
    data.abstract_texts = [record.abstract for record in records]


def _struct_cache_dir(dataset_name: str) -> str:
    cache_dir = os.path.join("datasets", "structural_cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _text_embed_cache_dir(dataset_name: str) -> str:
    cache_dir = os.path.join("datasets", "text_embed_cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _sanitize_model_tag(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(model_name))


def _text_embed_cache_path(dataset_name: str, model_name: str, embedding_dimensions: int | None) -> str:
    dim_tag = "default" if embedding_dimensions is None else str(int(embedding_dimensions))
    model_tag = _sanitize_model_tag(model_name)
    filename = f"text_embed_{model_tag}_dim{dim_tag}.pt"
    return os.path.join(_text_embed_cache_dir(dataset_name), filename)


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _struct_feature_cache_path(
    dataset_name: str,
    node2vec_dim: int,
    ppr_dim: int,
    laplacian_dim: int,
    community_dim: int,
) -> str:
    filename = (
        f"struct_features_n2v{int(node2vec_dim)}_ppr{int(ppr_dim)}_"
        f"lap{int(laplacian_dim)}_comm{int(community_dim)}.pt"
    )
    return os.path.join(_struct_cache_dir(dataset_name), filename)


def _build_node2vec_features(
    data,
    dataset_name: str,
    embedding_dim: int = 64,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 10,
    epochs: int = 30,
    lr: float = 0.01,
) -> torch.Tensor:
    """Build or load cached Node2Vec embeddings for structure-only teacher input."""
    cache_path = os.path.join(_struct_cache_dir(dataset_name), f"node2vec_dim{embedding_dim}.pt")
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))

    if os.path.exists(cache_path):
        payload = _safe_torch_load(cache_path)
        emb = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if (
            isinstance(emb, torch.Tensor)
            and emb.size(0) == num_nodes
            and emb.size(1) == embedding_dim
            and int(payload.get("num_edges", num_edges)) == num_edges
        ):
            print(f"[Structure] Loaded Node2Vec cache: {cache_path}")
            return emb.float()

    from torch_geometric.nn import Node2Vec

    print(f"[Structure] Building Node2Vec features (dim={embedding_dim}, epochs={epochs})...")
    node2vec = Node2Vec(
        data.edge_index.cpu(),
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    )
    loader = node2vec.loader(
        batch_size=min(256, max(64, num_nodes)),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)

    node2vec.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    emb = node2vec.embedding.weight.detach().cpu().float()
    emb = (emb - emb.mean(dim=0, keepdim=True)) / emb.std(dim=0, keepdim=True).clamp(min=1e-8)
    torch.save(
        {
            "embeddings": emb,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "dim": embedding_dim,
        },
        cache_path,
    )
    print(f"[Structure] Saved Node2Vec cache: {cache_path}")
    return emb


def _build_anchor_ppr_features(
    data,
    dataset_name: str,
    dim: int = 16,
    alpha: float = 0.15,
    steps: int = 20,
) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((int(data.num_nodes), 0), dtype=torch.float32)

    cache_path = os.path.join(_struct_cache_dir(dataset_name), f"ppr_anchor_dim{dim}.pt")
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    if os.path.exists(cache_path):
        payload = _safe_torch_load(cache_path)
        emb = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if (
            isinstance(emb, torch.Tensor)
            and emb.size(0) == num_nodes
            and emb.size(1) == dim
            and int(payload.get("num_edges", num_edges)) == num_edges
        ):
            print(f"[Structure] Loaded PPR-anchor cache: {cache_path}")
            return emb.float()

    edge_index = to_undirected(data.edge_index.cpu(), num_nodes=num_nodes)
    row, col = edge_index
    values = torch.ones(row.size(0), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (num_nodes, num_nodes)).coalesce()
    degree_vec = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    d_inv = 1.0 / degree_vec

    anchor_scores = degree_vec
    anchor_indices = torch.topk(anchor_scores, k=min(dim, num_nodes), largest=True).indices.tolist()
    features = []
    for anchor_idx in anchor_indices:
        restart = torch.zeros(num_nodes, dtype=torch.float32)
        restart[anchor_idx] = 1.0
        state = restart.clone()
        for _ in range(steps):
            propagated = torch.sparse.mm(adj, state.unsqueeze(-1)).squeeze(-1) * d_inv
            state = alpha * restart + (1.0 - alpha) * propagated
        features.append(state)

    ppr = torch.stack(features, dim=-1) if features else torch.zeros((num_nodes, 0), dtype=torch.float32)
    if ppr.size(1) < dim:
        pad = torch.zeros((num_nodes, dim - ppr.size(1)), dtype=torch.float32)
        ppr = torch.cat([ppr, pad], dim=-1)
    ppr = (ppr - ppr.mean(dim=0, keepdim=True)) / ppr.std(dim=0, keepdim=True).clamp(min=1e-8)
    torch.save({"embeddings": ppr, "num_nodes": num_nodes, "num_edges": num_edges, "dim": dim}, cache_path)
    print(f"[Structure] Saved PPR-anchor cache: {cache_path}")
    return ppr


def _build_laplacian_pe_features(data, dataset_name: str, dim: int = 8) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((int(data.num_nodes), 0), dtype=torch.float32)

    cache_path = os.path.join(_struct_cache_dir(dataset_name), f"laplacian_pe_dim{dim}.pt")
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    if os.path.exists(cache_path):
        payload = _safe_torch_load(cache_path)
        emb = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if (
            isinstance(emb, torch.Tensor)
            and emb.size(0) == num_nodes
            and emb.size(1) == dim
            and int(payload.get("num_edges", num_edges)) == num_edges
        ):
            print(f"[Structure] Loaded Laplacian PE cache: {cache_path}")
            return emb.float()

    if sp is None or eigsh is None:
        return torch.zeros((num_nodes, dim), dtype=torch.float32)

    edge_index = to_undirected(data.edge_index.cpu(), num_nodes=num_nodes)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    values = np.ones_like(row, dtype=np.float32)
    adj = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(np.clip(deg, 1.0, None), -0.5)
    d_mat = sp.diags(deg_inv_sqrt)
    lap = sp.eye(num_nodes, dtype=np.float32) - d_mat @ adj @ d_mat

    effective_dim = min(dim + 1, max(2, num_nodes - 1))
    try:
        _, vecs = eigsh(lap, k=effective_dim, which="SM")
        vecs = vecs[:, 1:effective_dim]
    except Exception:
        vecs = np.zeros((num_nodes, dim), dtype=np.float32)

    pe = torch.tensor(vecs, dtype=torch.float32)
    if pe.size(1) < dim:
        pe = torch.cat([pe, torch.zeros((num_nodes, dim - pe.size(1)), dtype=torch.float32)], dim=-1)
    pe = pe[:, :dim]
    pe = (pe - pe.mean(dim=0, keepdim=True)) / pe.std(dim=0, keepdim=True).clamp(min=1e-8)
    torch.save({"embeddings": pe, "num_nodes": num_nodes, "num_edges": num_edges, "dim": dim}, cache_path)
    print(f"[Structure] Saved Laplacian PE cache: {cache_path}")
    return pe


def _build_community_features(data, dataset_name: str, dim: int = 8) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((int(data.num_nodes), 0), dtype=torch.float32)

    cache_path = os.path.join(_struct_cache_dir(dataset_name), f"community_dim{dim}.pt")
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    if os.path.exists(cache_path):
        payload = _safe_torch_load(cache_path)
        emb = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if (
            isinstance(emb, torch.Tensor)
            and emb.size(0) == num_nodes
            and emb.size(1) == dim
            and int(payload.get("num_edges", num_edges)) == num_edges
        ):
            print(f"[Structure] Loaded community cache: {cache_path}")
            return emb.float()

    graph_nx = to_networkx(data, to_undirected=True)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph_nx))
    communities = sorted(communities, key=len, reverse=True)
    top_communities = communities[:dim]
    features = torch.zeros((num_nodes, dim), dtype=torch.float32)
    overflow_column = dim - 1
    for community_idx, nodes in enumerate(communities):
        target_column = community_idx if community_idx < dim - 1 else overflow_column
        node_list = list(nodes)
        features[node_list, target_column] = 1.0
    if len(top_communities) <= 1:
        return features
    return features


def build_structural_features(
    data,
    dataset_name: str,
    node2vec_dim: int = 64,
    ppr_dim: int = 16,
    laplacian_dim: int = 8,
    community_dim: int = 8,
) -> torch.Tensor:
    """Create stronger structure-only node features from graph topology."""
    num_nodes = int(data.num_nodes)
    feature_cache_path = _struct_feature_cache_path(
        dataset_name=dataset_name,
        node2vec_dim=node2vec_dim,
        ppr_dim=ppr_dim,
        laplacian_dim=laplacian_dim,
        community_dim=community_dim,
    )
    num_edges = int(data.edge_index.size(1))
    if os.path.exists(feature_cache_path):
        payload = _safe_torch_load(feature_cache_path)
        emb = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if (
            isinstance(emb, torch.Tensor)
            and emb.size(0) == num_nodes
            and int(payload.get("num_edges", num_edges)) == num_edges
        ):
            print(f"[Structure] Loaded structural feature cache: {feature_cache_path}")
            return emb.float()

    edge_index = data.edge_index.cpu()
    row, col = edge_index

    ones = torch.ones(row.size(0), dtype=torch.float32)
    in_degree = torch.zeros(num_nodes, dtype=torch.float32).scatter_add_(0, col, ones)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32).scatter_add_(0, row, ones)
    degree_mean = (in_degree + out_degree) / 2.0
    degree_norm = degree_mean / degree_mean.clamp(min=1.0).max()

    avg_neighbor_degree = torch.zeros(num_nodes, dtype=torch.float32)
    if row.numel() > 0:
        avg_neighbor_degree.scatter_add_(0, row, degree_mean[col])
        avg_neighbor_degree = avg_neighbor_degree / out_degree.clamp(min=1.0)

    graph_nx = to_networkx(data, to_undirected=True)
    clustering = nx.clustering(graph_nx)
    pagerank = nx.pagerank(graph_nx)

    clustering_tensor = torch.tensor(
        [clustering.get(node_id, 0.0) for node_id in range(num_nodes)],
        dtype=torch.float32,
    )
    pagerank_tensor = torch.tensor(
        [pagerank.get(node_id, 0.0) for node_id in range(num_nodes)],
        dtype=torch.float32,
    )
    base_features = torch.stack(
        [
            degree_mean,
            in_degree,
            out_degree,
            avg_neighbor_degree,
            degree_norm,
            clustering_tensor,
            pagerank_tensor,
        ],
        dim=-1,
    )
    base_features = (base_features - base_features.mean(dim=0, keepdim=True)) / base_features.std(
        dim=0, keepdim=True
    ).clamp(min=1e-8)

    feature_blocks = [base_features]
    if node2vec_dim > 0:
        feature_blocks.append(
            _build_node2vec_features(
                data=data,
                dataset_name=dataset_name,
                embedding_dim=node2vec_dim,
            )
        )
    if ppr_dim > 0:
        feature_blocks.append(_build_anchor_ppr_features(data=data, dataset_name=dataset_name, dim=ppr_dim))
    if laplacian_dim > 0:
        feature_blocks.append(_build_laplacian_pe_features(data=data, dataset_name=dataset_name, dim=laplacian_dim))
    if community_dim > 0:
        feature_blocks.append(_build_community_features(data=data, dataset_name=dataset_name, dim=community_dim))

    structural_x = torch.cat(feature_blocks, dim=-1)
    torch.save(
        {
            "embeddings": structural_x,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node2vec_dim": node2vec_dim,
            "ppr_dim": ppr_dim,
            "laplacian_dim": laplacian_dim,
            "community_dim": community_dim,
        },
        feature_cache_path,
    )
    print(f"[Structure] Saved structural feature cache: {feature_cache_path}")
    return structural_x


def load_dual_teacher_dataset(
    dataset_name: str,
    device: str = "cpu",
    sbert_model_name: str = "all-mpnet-base-v2",
    sbert_embedding_dimensions: int | None = None,
    struct_node2vec_dim: int = 64,
    struct_ppr_dim: int = 0,
    struct_laplacian_dim: int = 0,
    struct_community_dim: int = 0,
) -> Tuple[object, str]:
    """Load a TAG dataset and replace node features with SBERT embeddings."""
    canonical_name, dataset_path = resolve_dataset_path(dataset_name)
    data = safe_torch_load(dataset_path, map_location="cpu")

    ensure_text_fields(data)
    data.label_names = resolve_label_names(data, canonical_name)

    text_cache_path = _text_embed_cache_path(
        dataset_name=canonical_name,
        model_name=sbert_model_name,
        embedding_dimensions=sbert_embedding_dimensions,
    )
    force_rebuild_text_cache = os.getenv("FORCE_REBUILD_TEXT_EMBED_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    embedding_usage_log = os.getenv("OPENAI_EMBEDDING_USAGE_LOG", "").strip()
    cached_text = None
    if not force_rebuild_text_cache and os.path.exists(text_cache_path):
        cached_text = _safe_torch_load(text_cache_path)
    if (
        isinstance(cached_text, dict)
        and "embeddings" in cached_text
        and isinstance(cached_text["embeddings"], torch.Tensor)
        and int(cached_text.get("num_nodes", -1)) == int(data.num_nodes)
    ):
        data.x = cached_text["embeddings"].to(dtype=torch.float32, device="cpu").clone()
        print(f"[TextEmbed] Loaded cache: {text_cache_path}")
        if embedding_usage_log:
            os.makedirs(os.path.dirname(embedding_usage_log), exist_ok=True)
            with open(embedding_usage_log, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": canonical_name,
                        "model": sbert_model_name,
                        "embedding_dimensions": sbert_embedding_dimensions,
                        "num_nodes": int(data.num_nodes),
                        "cache_path": text_cache_path,
                        "cache_used": True,
                        "force_rebuild_text_cache": False,
                        "requests": 0,
                        "input_count": 0,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
    else:
        if force_rebuild_text_cache and os.path.exists(text_cache_path):
            print(f"[TextEmbed] FORCE_REBUILD_TEXT_EMBED_CACHE=true; ignoring cache: {text_cache_path}")
        embedder = get_embedder(
            model_name=sbert_model_name,
            device=device,
            embedding_dimensions=sbert_embedding_dimensions,
        )
        if hasattr(embedder, "reset_usage_stats"):
            embedder.reset_usage_stats()
        data.x = embedder.embed_batch(data.raw_texts).detach().to(dtype=torch.float32, device="cpu").clone()
        torch.save(
            {
                "embeddings": data.x,
                "num_nodes": int(data.num_nodes),
                "model_name": sbert_model_name,
                "embedding_dimensions": sbert_embedding_dimensions,
            },
            text_cache_path,
        )
        print(f"[TextEmbed] Saved cache: {text_cache_path}")
        if embedding_usage_log:
            usage_stats = embedder.get_usage_stats() if hasattr(embedder, "get_usage_stats") else {}
            usage_stats.pop("prompt_tokens", None)
            usage_stats.pop("completion_tokens", None)
            usage_stats.pop("total_tokens", None)
            usage_stats.update(
                {
                    "dataset": canonical_name,
                    "model": sbert_model_name,
                    "embedding_dimensions": sbert_embedding_dimensions,
                    "num_nodes": int(data.num_nodes),
                    "cache_path": text_cache_path,
                    "cache_used": False,
                    "force_rebuild_text_cache": bool(force_rebuild_text_cache),
                }
            )
            os.makedirs(os.path.dirname(embedding_usage_log), exist_ok=True)
            with open(embedding_usage_log, "w", encoding="utf-8") as handle:
                json.dump(usage_stats, handle, indent=2, ensure_ascii=False)
    data.sbert_dim = int(data.x.size(-1))
    if struct_node2vec_dim <= 0 and struct_ppr_dim <= 0 and struct_laplacian_dim <= 0 and struct_community_dim <= 0:
        data.structural_x = data.x.clone()
        data.structural_dim = int(data.sbert_dim)
        print("[Structure] Using text embeddings only for GNN Teacher inputs.")
    else:
        data.structural_x = build_structural_features(
            data,
            dataset_name=canonical_name,
            node2vec_dim=struct_node2vec_dim,
            ppr_dim=struct_ppr_dim,
            laplacian_dim=struct_laplacian_dim,
            community_dim=struct_community_dim,
        )
        data.structural_dim = int(data.structural_x.size(-1))
    return data, canonical_name


def apply_dual_teacher_split(
    data,
    dataset_name: str,
    query_ratio: float,
    k_shot: int,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    query_selection_method: str = "random",
    query_selection_feature_source: str = "structural",
    query_selection_num_clusters: int = 0,
    teacher_mode: str = "dual",
):
    """Apply the train/val/test + query/unqueried masks required by the framework (zero-shot)."""
    split_data = create_few_shot_split(
        data=data,
        dataset_name=dataset_name,
        query_ratio=query_ratio,
        k_shot=k_shot,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        query_selection_method=query_selection_method,
        query_selection_feature_source=query_selection_feature_source,
        query_selection_num_clusters=query_selection_num_clusters,
        teacher_mode=teacher_mode,
    )
    return split_data
