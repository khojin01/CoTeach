"""GNN-teacher module for SSL + k-shot multitask learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from models.self_supervised_teacher import BGRLTeacher, create_ssl_teacher


@dataclass
class GNNTeacherOutputs:
    embeddings: torch.Tensor


class MultiTaskGNNTeacher(nn.Module):
    """SSL teacher that learns graph structure and class boundaries together."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        ssl_method: str = "dgi",
        backbone: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
        lambda_ce_kshot: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.lambda_ce_kshot = float(lambda_ce_kshot)
        self.ssl_method = ssl_method
        self.teacher = create_ssl_teacher(
            method=ssl_method,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            backbone=backbone,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.teacher.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _as_structural_graph(self, data: Data) -> Data:
        graph = data.clone()
        if not hasattr(graph, "structural_x"):
            raise AttributeError("`structural_x` is required for the structure-only GNN teacher.")
        graph.x = graph.structural_x
        return graph.to(self.device)

    def _encode_with_grad(self, graph: Data) -> torch.Tensor:
        if hasattr(self.teacher, "online_encoder"):
            embeddings = self.teacher.online_encoder(graph.x, graph.edge_index)
        elif hasattr(self.teacher, "encoder"):
            embeddings = self.teacher.encoder(graph.x, graph.edge_index)
        else:
            raise AttributeError(f"Unsupported SSL teacher for gradient encoding: {type(self.teacher).__name__}")
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, data: Data) -> GNNTeacherOutputs:
        graph = self._as_structural_graph(data)
        embeddings = self.teacher.get_embeddings(graph)
        return GNNTeacherOutputs(
            embeddings=embeddings,
        )

    def fit(self, data: Data, epochs: int = 300, verbose: bool = True) -> None:
        def _run_fit(graph: Data, use_amp: bool) -> None:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            for epoch in range(epochs):
                self.teacher.train()
                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    ssl_loss = self.teacher.compute_loss(graph)
                    loss = ssl_loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if isinstance(self.teacher, BGRLTeacher):
                    self.teacher.update_target_encoder()

                if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                    print(
                        f"[GNN Teacher] Epoch {epoch + 1}/{epochs} | "
                        f"Total={loss.item():.4f} | SSL={ssl_loss.item():.4f}"
                    )

        graph = self._as_structural_graph(data)
        use_amp = self.device.type == "cuda"
        try:
            _run_fit(graph=graph, use_amp=use_amp)
            return
        except torch.OutOfMemoryError:
            if self.device.type != "cuda":
                raise
            if verbose:
                print("[GNN Teacher] CUDA OOM detected. Falling back to CPU training for stability.")
            torch.cuda.empty_cache()

        # CPU fallback: keep the run alive even on large graphs/high dimensions.
        self.device = torch.device("cpu")
        self.teacher = self.teacher.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.teacher.parameters()),
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        cpu_graph = data.clone()
        cpu_graph.x = cpu_graph.structural_x
        cpu_graph = cpu_graph.to(self.device)
        _run_fit(graph=cpu_graph, use_amp=False)

    @torch.no_grad()
    def infer(self, data: Data) -> GNNTeacherOutputs:
        self.teacher.eval()
        return self.forward(data)
