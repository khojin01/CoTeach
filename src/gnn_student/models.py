"""Student-side GNN modules for dual-teacher distillation."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """Shared Graph encoder block used by single-view and dual-view students."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        backbone: str = "graphsage",
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "graphsage":
            from torch_geometric.nn import SAGEConv

            def make_conv(in_dim: int, out_dim: int):
                return SAGEConv(in_dim, out_dim)

        elif backbone == "gcn":
            from torch_geometric.nn import GCNConv
            def make_conv(in_dim: int, out_dim: int):
                return GCNConv(in_dim, out_dim)
        elif backbone == "gat":
            from torch_geometric.nn import GATConv

            # Keep output dimensions stable across layers for drop-in parity.
            def make_conv(in_dim: int, out_dim: int):
                return GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Choose from 'graphsage', 'gcn', 'gat'."
            )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        if num_layers <= 1:
            self.convs.append(make_conv(input_dim, embedding_dim))
        else:
            self.convs.append(make_conv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(make_conv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(make_conv(hidden_dim, embedding_dim))

    def forward(self, x, edge_index):
        last_idx = len(self.convs) - 1
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != last_idx:
                x = self.bns[idx](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def set_bn_eval(self):
        for bn in self.bns:
            bn.eval()


class StudentGNN(nn.Module):
    """Single-view student kept for compatibility when structural view is disabled."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        alignment_dim: int = 768,  # SBERT dimension
        backbone: str = "graphsage",
    ):
        super().__init__()
        # Increase model capacity
        hidden_dim_expanded = hidden_dim * 2 
        
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim_expanded,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            backbone=backbone,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_expanded),
            nn.BatchNorm1d(hidden_dim_expanded),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_expanded, num_classes),
        )
        # Head for Feature Alignment with LLM Reasoning (CoT)
        self.alignment_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_expanded),
            nn.BatchNorm1d(hidden_dim_expanded),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_expanded, alignment_dim)
        )
        self.runtime_mode = "semantic_only"

    def set_runtime_mode(self, mode: str):
        self.runtime_mode = str(mode)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def forward(self, data, return_details: bool = False):
        embeddings = self.encode(data.x, data.edge_index)
        logits = self.classifier(embeddings)
        alignment_emb = self.alignment_head(embeddings)
        
        if not return_details:
            return logits
        
        gate_alpha = torch.ones(embeddings.size(0), dtype=embeddings.dtype, device=embeddings.device)
        return {
            "logits": logits,
            "alignment_embeddings": alignment_emb,
            "fused_embeddings": embeddings,
            "semantic_embeddings": embeddings,
            "structural_embeddings": embeddings,
            "gate_alpha": gate_alpha,
        }

    def get_embeddings(self, data, view: str = "fused"):
        outputs = self.forward(data, return_details=True)
        key = {
            "fused": "fused_embeddings",
            "semantic": "semantic_embeddings",
            "structural": "structural_embeddings",
        }.get(view, "fused_embeddings")
        embeddings = outputs[key]
        return F.normalize(embeddings, p=2, dim=-1)

    def get_semantic_embeddings(self, data):
        return self.get_embeddings(data, view="semantic")

    def get_structural_embeddings(self, data):
        return self.get_embeddings(data, view="structural")

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            yield p

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder_parameters():
            p.requires_grad = trainable

    def set_encoder_bn_eval(self):
        self.encoder.set_bn_eval()


class DualViewStudentGNN(nn.Module):
    """Student with semantic and structural branches plus node-wise fusion gate."""

    def __init__(
        self,
        semantic_input_dim: int,
        structural_input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        structural_feature_fn: Callable,
        num_layers: int = 2,
        dropout: float = 0.5,
        alignment_dim: int = 768,  # SBERT dimension
    ):
        super().__init__()
        self.semantic_encoder = GraphEncoder(
            input_dim=semantic_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.structural_encoder = GraphEncoder(
            input_dim=structural_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        gate_hidden_dim = max(32, embedding_dim // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(min(0.2, dropout)),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, 0.8)

        self.fusion_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.ModuleDict(
            {
                "semantic": nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                ),
                "structural": nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                ),
            }
        )
        
        # Head for Feature Alignment with LLM Reasoning (CoT)
        self.alignment_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alignment_dim)
        )

        self.structural_feature_fn = structural_feature_fn
        self.runtime_mode = "adaptive"

    def set_runtime_mode(self, mode: str):
        mode = str(mode)
        if mode not in {"adaptive", "semantic_only", "structural_only"}:
            raise ValueError(f"Unknown student runtime mode: {mode}")
        self.runtime_mode = mode

    def _encode_semantic(self, data):
        return self.semantic_encoder(data.x, data.edge_index)

    def _encode_structural(self, data):
        structural_x = self.structural_feature_fn(data)
        return self.structural_encoder(structural_x, data.edge_index)

    def _fuse_embeddings(self, semantic_embeddings: torch.Tensor, structural_embeddings: torch.Tensor):
        if self.runtime_mode == "semantic_only":
            alpha = torch.ones(
                semantic_embeddings.size(0),
                dtype=semantic_embeddings.dtype,
                device=semantic_embeddings.device,
            )
            fused_embeddings = semantic_embeddings
        elif self.runtime_mode == "structural_only":
            alpha = torch.zeros(
                structural_embeddings.size(0),
                dtype=structural_embeddings.dtype,
                device=structural_embeddings.device,
            )
            fused_embeddings = structural_embeddings
        else:
            gate_input = torch.cat(
                [
                    semantic_embeddings,
                    structural_embeddings,
                    torch.abs(semantic_embeddings - structural_embeddings),
                ],
                dim=-1,
            )
            alpha = torch.sigmoid(self.gate_mlp(gate_input).squeeze(-1))
            fused_embeddings = (
                alpha.unsqueeze(-1) * semantic_embeddings
                + (1.0 - alpha).unsqueeze(-1) * structural_embeddings
            )
            fused_embeddings = self.fusion_norm(fused_embeddings)
        return fused_embeddings, alpha

    def forward(self, data, return_details: bool = False):
        if self.runtime_mode == "semantic_only":
            semantic_embeddings = self._encode_semantic(data)
            structural_embeddings = semantic_embeddings
        elif self.runtime_mode == "structural_only":
            structural_embeddings = self._encode_structural(data)
            semantic_embeddings = structural_embeddings
        else:
            semantic_embeddings = self._encode_semantic(data)
            structural_embeddings = self._encode_structural(data)

        fused_embeddings, gate_alpha = self._fuse_embeddings(
            semantic_embeddings=semantic_embeddings,
            structural_embeddings=structural_embeddings,
        )
        semantic_logits = self.classifier["semantic"](semantic_embeddings)
        structural_logits = self.classifier["structural"](structural_embeddings)
        
        # Alignment embedding from fused view
        alignment_emb = self.alignment_head(fused_embeddings)
        
        if self.runtime_mode == "semantic_only":
            logits = semantic_logits
        elif self.runtime_mode == "structural_only":
            logits = structural_logits
        else:
            logits = (
                gate_alpha.unsqueeze(-1) * semantic_logits
                + (1.0 - gate_alpha).unsqueeze(-1) * structural_logits
            )
        if not return_details:
            return logits
        return {
            "logits": logits,
            "alignment_embeddings": alignment_emb,
            "fused_embeddings": fused_embeddings,
            "semantic_embeddings": semantic_embeddings,
            "structural_embeddings": structural_embeddings,
            "semantic_logits": semantic_logits,
            "structural_logits": structural_logits,
            "gate_alpha": gate_alpha,
        }

    def get_embeddings(self, data, view: str = "fused"):
        outputs = self.forward(data, return_details=True)
        key = {
            "fused": "fused_embeddings",
            "semantic": "semantic_embeddings",
            "structural": "structural_embeddings",
        }.get(view, "fused_embeddings")
        embeddings = outputs[key]
        return F.normalize(embeddings, p=2, dim=-1)

    def get_semantic_embeddings(self, data):
        return self.get_embeddings(data, view="semantic")

    def get_structural_embeddings(self, data):
        return self.get_embeddings(data, view="structural")

    def encoder_parameters(self):
        for module in [
            self.semantic_encoder,
            self.structural_encoder,
            self.gate_mlp,
            self.fusion_norm,
        ]:
            for p in module.parameters():
                yield p

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder_parameters():
            p.requires_grad = trainable

    def set_encoder_bn_eval(self):
        self.semantic_encoder.set_bn_eval()
        self.structural_encoder.set_bn_eval()
