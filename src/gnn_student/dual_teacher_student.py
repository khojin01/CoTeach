"""Student GNN used for dual-teacher distillation."""

from __future__ import annotations

import torch
import torch.nn as nn

from gnn_student.models import GraphEncoder


class DistillStudentGNN(nn.Module):
    """Student encoder with projection head for structural alignment and classifier head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        alignment_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.5,
        backbone: str = "graphsage",
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            backbone=backbone,
        )
        self.logit_head = nn.Linear(hidden_dim, num_classes)
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, alignment_dim),
        )

    def forward(self, data):
        hidden = self.encoder(data.x, data.edge_index)
        logits = self.logit_head(hidden)
        projected_embeddings = self.alignment_head(hidden)
        return {
            "hidden": hidden,
            "logits": logits,
            "projected_embeddings": projected_embeddings,
        }
