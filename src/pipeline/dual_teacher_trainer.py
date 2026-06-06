"""Integrated trainer for dual-teacher distillation."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from gnn_student.dual_teacher_student import DistillStudentGNN
from gnn_student.losses import SemanticClassificationLoss, SoftProbDistillationLoss
from gnn_teacher.dual_teacher_gnn_teacher import MultiTaskGNNTeacher
from llm_teacher.dual_teacher_llm import DualTeacherLLM


class DualTeacherTrainer:
    """Orchestrates LLM Teacher + GNN Teacher + Student distillation."""

    def __init__(
        self,
        semantic_input_dim: int,
        structural_input_dim: int,
        num_classes: int,
        ssl_method: str,
        student_backbone: str,
        alpha: float,
        gnn_teacher_hidden_dim: int,
        gnn_teacher_embedding_dim: int,
        student_hidden_dim: int,
        student_alignment_dim: int,
        gnn_teacher_lr: float,
        student_lr: float,
        weight_decay: float,
        llm_model_name: str,
        llm_temperature: float,
        llm_max_output_tokens: int,
        device: str,
        distill_temperature: float = 1.0,
        gnn_logit_epochs: int = 200,
        gnn_logit_lr: float = 1e-3,
        gnn_logit_hidden_dim: int = 256,
        gnn_confidence_mode: str = "margin",
        gnn_confidence_threshold: float = float("-inf"),
        gnn_confidence_threshold_mode: str = "absolute",
        gnn_confidence_quantile: float = 0.5,
        llm_confidence_threshold: float = 0.0,
        gnn_priority_llm_fallback: bool = False,
        llm_query_from_gnn_uncertain_pool: bool = False,
        teacher_mode: str = "dual",
    ):
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.num_classes = int(num_classes)

        self.gnn_logit_epochs = int(gnn_logit_epochs)
        self.gnn_logit_lr = float(gnn_logit_lr)
        self.gnn_logit_hidden_dim = int(gnn_logit_hidden_dim)

        self.gnn_confidence_mode = str(gnn_confidence_mode)
        self.gnn_confidence_threshold = float(gnn_confidence_threshold)
        self.gnn_confidence_threshold_mode = str(gnn_confidence_threshold_mode)
        self.gnn_confidence_quantile = float(gnn_confidence_quantile)
        if self.gnn_confidence_threshold_mode not in {"absolute", "quantile"}:
            raise ValueError("gnn_confidence_threshold_mode must be 'absolute' or 'quantile'.")
        if not 0.0 <= self.gnn_confidence_quantile <= 1.0:
            raise ValueError("gnn_confidence_quantile must be in [0, 1].")
        self.gnn_confidence_threshold_used = float(self.gnn_confidence_threshold)
        self.llm_confidence_threshold = float(llm_confidence_threshold)
        self.gnn_priority_llm_fallback = bool(gnn_priority_llm_fallback)
        self.llm_query_from_gnn_uncertain_pool = bool(llm_query_from_gnn_uncertain_pool)
        self.teacher_mode = str(teacher_mode)

        self.gnn_teacher = MultiTaskGNNTeacher(
            input_dim=structural_input_dim,
            hidden_dim=gnn_teacher_hidden_dim,
            embedding_dim=gnn_teacher_embedding_dim,
            num_classes=num_classes,
            ssl_method=ssl_method,
            lambda_ce_kshot=0.0,
            lr=gnn_teacher_lr,
            weight_decay=weight_decay,
            device=device,
        )

        self.student = DistillStudentGNN(
            input_dim=semantic_input_dim,
            hidden_dim=student_hidden_dim,
            num_classes=num_classes,
            alignment_dim=student_alignment_dim,
            backbone=student_backbone,
        ).to(self.device)
        self.student_optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=student_lr,
            weight_decay=weight_decay,
        )

        self.distill_prob_loss = SoftProbDistillationLoss(temperature=distill_temperature)
        self.semantic_loss = SemanticClassificationLoss()

        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.llm_max_output_tokens = llm_max_output_tokens
        self.llm_embedder_device = str(device)
        self.gnn_teacher_cache_dir = os.environ.get("DUAL_TEACHER_GNN_CACHE_DIR", "")

    def _resolve_gnn_confidence_threshold(self, gnn_conf: torch.Tensor, base_mask: torch.Tensor) -> float:
        if self.gnn_confidence_threshold_mode == "absolute":
            threshold = float(self.gnn_confidence_threshold)
        else:
            valid_mask = base_mask.bool() & torch.isfinite(gnn_conf)
            if valid_mask.any():
                values = gnn_conf[valid_mask].float()
                q = torch.tensor(self.gnn_confidence_quantile, device=values.device)
                threshold = float(torch.quantile(values, q).item())
            else:
                threshold = float("inf")
        self.gnn_confidence_threshold_used = threshold
        return threshold

    def _gnn_teacher_cache_path(self, dataset_name: str, seed: int, data) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        cache_root = Path(self.gnn_teacher_cache_dir) if self.gnn_teacher_cache_dir else (repo_root / "datasets" / "gnn_teacher_cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        num_nodes = int(data.x.size(0))
        in_dim = int(getattr(data, "structural_dim", data.structural_x.size(-1)))
        emb_dim = int(self.gnn_teacher.teacher.embedding_dim)
        hidden_dim = int(self.gnn_teacher.teacher.hidden_dim)
        ssl_name = str(self.gnn_teacher.ssl_method)
        train_mask = data.train_mask.detach().cpu().bool() if hasattr(data, "train_mask") else torch.zeros(num_nodes, dtype=torch.bool)
        k_shot_mask = data.k_shot_mask.detach().cpu().bool() if hasattr(data, "k_shot_mask") else torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = data.val_mask.detach().cpu().bool() if hasattr(data, "val_mask") else torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = data.test_mask.detach().cpu().bool() if hasattr(data, "test_mask") else torch.zeros(num_nodes, dtype=torch.bool)
        split_sig = torch.stack(
            [train_mask.to(torch.uint8), k_shot_mask.to(torch.uint8), val_mask.to(torch.uint8), test_mask.to(torch.uint8)],
            dim=0,
        ).numpy().tobytes()
        split_hash = hashlib.md5(split_sig).hexdigest()[:10]
        file_name = (
            f"{dataset_name}_seed{seed}_{ssl_name}_n{num_nodes}_in{in_dim}_h{hidden_dim}_e{emb_dim}"
            f"_k{int(k_shot_mask.sum().item())}_split{split_hash}.pt"
        )
        return str((cache_root / file_name).resolve())

    def _empty_llm_outputs(self, data) -> Dict[str, torch.Tensor]:
        num_nodes = int(data.x.size(0))
        probs = torch.full(
            (num_nodes, self.num_classes),
            1.0 / float(self.num_classes),
            device=self.device,
        )
        return {
            "predictions": torch.full((num_nodes,), -1, dtype=torch.long, device=self.device),
            "confidences": torch.zeros(num_nodes, dtype=torch.float, device=self.device),
            "probs": probs,
            "valid_query_mask": torch.zeros(num_nodes, dtype=torch.bool, device=self.device),
            "valid_prob_mask": torch.zeros(num_nodes, dtype=torch.bool, device=self.device),
        }

    def _empty_gnn_outputs(self, data) -> Dict[str, torch.Tensor]:
        num_nodes = int(data.x.size(0))
        probs = torch.full(
            (num_nodes, self.num_classes),
            1.0 / float(self.num_classes),
            device=self.device,
        )
        empty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        return {
            "logits": torch.zeros((num_nodes, self.num_classes), dtype=torch.float, device=self.device),
            "probs": probs,
            "predictions": torch.zeros(num_nodes, dtype=torch.long, device=self.device),
            "confidences": torch.full((num_nodes,), float("-inf"), dtype=torch.float, device=self.device),
            "head_train_mask": empty_mask.clone(),
            "head_holdout_mask": empty_mask,
            "head_train_acc": torch.tensor(0.0, device=self.device),
            "head_holdout_acc": torch.tensor(0.0, device=self.device),
            "head_train_source": "disabled",
        }

    def _split_train_for_gnn_head(self, data, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        train_mask = data.train_mask.to(self.device).bool()
        if hasattr(data, "k_shot_mask"):
            k_shot_mask = data.k_shot_mask.to(self.device).bool()
        else:
            k_shot_mask = torch.zeros_like(train_mask)

        # Few-shot priority: use exactly the shared k-shot nodes as GNN head train set.
        # This keeps GNN Teacher supervision aligned with LLM Teacher few-shot exemplars.
        head_train_mask = train_mask & k_shot_mask
        if int(head_train_mask.sum().item()) > 0:
            head_holdout_mask = train_mask & ~head_train_mask
            return head_train_mask, head_holdout_mask

        # Fallback (k_shot==0): use all train nodes for the GNN head.
        train_indices = torch.where(train_mask)[0]
        if train_indices.numel() == 0:
            empty = torch.zeros_like(train_mask)
            return empty, empty

        head_train_mask = torch.zeros_like(train_mask)
        head_holdout_mask = torch.zeros_like(train_mask)
        head_train_mask[train_indices] = True
        return head_train_mask, head_holdout_mask

    def _gnn_confidence_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        mode = str(self.gnn_confidence_mode or "top1_logit")
        if mode == "top1_logit":
            return logits.max(dim=-1).values
        if mode == "margin":
            top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
            if top2.size(-1) == 1:
                return top2[:, 0]
            return top2[:, 0] - top2[:, 1]

        # Probability-space confidence (recommended; bounded in [0,1])
        probs = F.softmax(logits, dim=-1)
        if mode == "top1_prob":
            return probs.max(dim=-1).values
        if mode == "margin_prob":
            top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
            if top2.size(-1) == 1:
                return top2[:, 0]
            return top2[:, 0] - top2[:, 1]
        if mode == "entropy":
            # Normalized confidence: 1 - H(p)/log(C) in [0,1]
            eps = 1e-8
            ent = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
            denom = float(torch.log(torch.tensor(probs.size(-1), device=logits.device)).clamp_min(eps).item())
            return 1.0 - (ent / max(denom, eps))

        raise ValueError(f"Unsupported gnn_confidence_mode: {mode}")

    def _train_structural_logit_head(
        self,
        fixed_structural_map: torch.Tensor,
        data,
        seed: int,
    ) -> Dict[str, torch.Tensor]:
        inputs = fixed_structural_map.detach()
        labels = data.y.to(self.device).long()
        head_train_mask, head_holdout_mask = self._split_train_for_gnn_head(data, seed=seed)

        head = torch.nn.Sequential(
            torch.nn.Linear(inputs.size(-1), self.gnn_logit_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.gnn_logit_hidden_dim, self.num_classes),
        ).to(self.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=self.gnn_logit_lr)

        if head_train_mask.any():
            for _ in range(self.gnn_logit_epochs):
                optimizer.zero_grad()
                logits = head(inputs)
                loss = F.cross_entropy(logits[head_train_mask], labels[head_train_mask])
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits = head(inputs)
            predictions = logits.argmax(dim=-1)
            confidences = self._gnn_confidence_from_logits(logits)
            holdout_acc = 0.0
            if head_holdout_mask.any():
                holdout_acc = float((predictions[head_holdout_mask] == labels[head_holdout_mask]).float().mean().item())
            train_acc = 0.0
            if head_train_mask.any():
                train_acc = float((predictions[head_train_mask] == labels[head_train_mask]).float().mean().item())

        source = "k_shot_mask" if hasattr(data, "k_shot_mask") and int((data.k_shot_mask.to(self.device).bool() & data.train_mask.to(self.device).bool()).sum().item()) > 0 else "train_random_split"
        print(
            f"[GNN Head] Train source={source} | "
            f"train_nodes={int(head_train_mask.sum().item())} | "
            f"holdout_nodes={int(head_holdout_mask.sum().item())}"
        )

        return {
            "logits": logits.detach(),
            "probs": F.softmax(logits.detach(), dim=-1),
            "predictions": predictions.detach(),
            "confidences": confidences.detach(),
            "head_train_mask": head_train_mask.detach(),
            "head_holdout_mask": head_holdout_mask.detach(),
            "head_train_acc": torch.tensor(train_acc, device=self.device),
            "head_holdout_acc": torch.tensor(holdout_acc, device=self.device),
            "head_train_source": source,
        }

    def _teacher_masks(
        self,
        data,
        llm_outputs: Dict[str, torch.Tensor],
        gnn_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_mask = data.train_mask.to(self.device).bool()
        query_mask = data.query_mask.to(self.device).bool()
        llm_valid = llm_outputs["valid_query_mask"].to(self.device).bool()
        llm_prob_valid = llm_outputs["valid_prob_mask"].to(self.device).bool()
        llm_conf = llm_outputs["confidences"].to(self.device).float()
        llm_keep = train_mask & query_mask & llm_valid & llm_prob_valid & (llm_conf >= self.llm_confidence_threshold)

        gnn_conf = gnn_outputs["confidences"].to(self.device).float()
        gnn_threshold = self._resolve_gnn_confidence_threshold(gnn_conf, train_mask)
        gnn_keep = train_mask & torch.isfinite(gnn_conf) & (gnn_conf >= gnn_threshold)

        # Priority routing (reverse): let GNN supervise where it is confident; use LLM as fallback
        # only on the remaining nodes. This removes overlap by construction.
        if self.gnn_priority_llm_fallback and self.teacher_mode == "dual":
            llm_keep = llm_keep & ~gnn_keep
        if self.teacher_mode == "gnn_only":
            llm_keep = torch.zeros_like(llm_keep)
        elif self.teacher_mode == "llm_only":
            gnn_keep = torch.zeros_like(gnn_keep)
        return llm_keep, gnn_keep

    @staticmethod
    def _append_confidence_stats(
        stats: Dict[str, float],
        prefix: str,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        mask = mask.bool()
        count = int(mask.sum().item())
        stats[f"{prefix}_count"] = count
        if count == 0:
            for k in ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"):
                stats[f"{prefix}_{k}"] = 0.0
            return
        selected = values[mask].float()
        quantiles = torch.quantile(
            selected,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=selected.device),
        )
        stats[f"{prefix}_mean"] = float(selected.mean().item())
        stats[f"{prefix}_std"] = float(selected.std(unbiased=False).item())
        stats[f"{prefix}_min"] = float(selected.min().item())
        stats[f"{prefix}_max"] = float(selected.max().item())
        stats[f"{prefix}_p10"] = float(quantiles[0].item())
        stats[f"{prefix}_p25"] = float(quantiles[1].item())
        stats[f"{prefix}_p50"] = float(quantiles[2].item())
        stats[f"{prefix}_p75"] = float(quantiles[3].item())
        stats[f"{prefix}_p90"] = float(quantiles[4].item())

    def _teacher_supervision_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_mask: torch.Tensor,
        use_soft_prob: bool = True,
        use_hard_ce: bool = True,
        sample_weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=student_logits.device)
        loss_ce = (
            self.semantic_loss(student_logits, teacher_labels, teacher_mask, sample_weights=sample_weights)
            if use_hard_ce
            else zero
        )
        if use_soft_prob:
            loss_prob = self.distill_prob_loss(student_logits, teacher_probs, teacher_mask, sample_weights=sample_weights)
            return loss_prob + loss_ce, loss_prob, loss_ce
        else:
            # Hard label only: no soft probability distillation
            return loss_ce, zero, loss_ce

    @torch.no_grad()
    def _compute_teacher_diagnostics(
        self,
        data,
        llm_outputs: Dict[str, torch.Tensor],
        gnn_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        labels = data.y.to(self.device)
        llm_mask, gnn_mask = self._teacher_masks(data, llm_outputs, gnn_outputs)
        query_mask = data.query_mask.to(self.device).bool()
        train_mask = data.train_mask.to(self.device).bool()
        all_mask = torch.ones_like(train_mask)

        llm_pred = llm_outputs["predictions"].to(self.device).long().clamp_min(0)
        llm_valid = llm_outputs.get("valid_query_mask", torch.zeros_like(query_mask)).to(self.device).bool()
        llm_conf = llm_outputs.get("confidences", torch.zeros_like(labels).float()).to(self.device).float()
        gnn_pred = gnn_outputs["predictions"].to(self.device).long()

        def _masked_acc(pred: torch.Tensor, mask: torch.Tensor) -> float:
            if not mask.any():
                return 0.0
            return float((pred[mask] == labels[mask]).float().mean().item())

        gnn_conf = gnn_outputs["confidences"].to(self.device).float()

        gnn_conf_all = gnn_conf[all_mask]
        gnn_conf_quartiles = torch.quantile(
            gnn_conf_all,
            torch.tensor([0.25, 0.50, 0.75], device=gnn_conf_all.device),
        )

        # LLM confidence threshold curve on queried nodes (parsed only)
        llm_query_base = query_mask & llm_valid
        llm_curve_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
        llm_curve_counts: Dict[str, int] = {}
        llm_curve_accs: Dict[str, float] = {}
        for t in llm_curve_thresholds:
            m = llm_query_base & (llm_conf >= float(t))
            key = f"{t:.2f}"
            llm_curve_counts[key] = int(m.sum().item())
            llm_curve_accs[key] = _masked_acc(llm_pred, m) if m.any() else 0.0

        results = {
            "llm_query_total_count": int(query_mask.sum().item()),
            "llm_kept_query_count": int(llm_mask.sum().item()),
            "gnn_kept_train_count": int(gnn_mask.sum().item()),
            "llm_kept_query_acc": _masked_acc(llm_pred, llm_mask),
            "gnn_kept_train_acc": _masked_acc(gnn_pred, gnn_mask),
            "gnn_mlp_test_acc": float(gnn_outputs["head_holdout_acc"].item()),
            "gnn_head_train_source": str(gnn_outputs.get("head_train_source", "unknown")),
            "gnn_head_train_node_count": int(gnn_outputs["head_train_mask"].sum().item()),
            "gnn_head_holdout_node_count": int(gnn_outputs["head_holdout_mask"].sum().item()),
            "gnn_confidence_threshold_mode": self.gnn_confidence_threshold_mode,
            "gnn_confidence_threshold_configured": float(self.gnn_confidence_threshold),
            "gnn_confidence_quantile": float(self.gnn_confidence_quantile),
            "gnn_confidence_threshold_used": float(self.gnn_confidence_threshold_used),
            "gnn_conf_all_q25": float(gnn_conf_quartiles[0].item()),
            "gnn_conf_all_q50": float(gnn_conf_quartiles[1].item()),
            "gnn_conf_all_q75": float(gnn_conf_quartiles[2].item()),
            # LLM confidence curve (queried nodes)
            "llm_query_conf_curve_thresholds": llm_curve_thresholds,
            "llm_query_conf_curve_counts": llm_curve_counts,
            "llm_query_conf_curve_accs": llm_curve_accs,
        }
        return results

    def _student_epoch(
        self,
        data,
        llm_outputs: Dict[str, torch.Tensor],
        gnn_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.student.train()
        self.student_optimizer.zero_grad()

        graph = data.to(self.device)
        logits = self.student(graph)["logits"]

        llm_mask, gnn_mask = self._teacher_masks(data, llm_outputs, gnn_outputs)
        llm_probs = llm_outputs["probs"].to(self.device)
        llm_pred = llm_outputs["predictions"].to(self.device).long().clamp_min(0)
        gnn_probs = gnn_outputs["probs"].to(self.device)
        gnn_pred = gnn_outputs["predictions"].to(self.device).long()
        gnn_conf = gnn_outputs.get("confidences", torch.zeros_like(gnn_pred).float()).to(self.device).float()

        # GNN-priority routing: GNN supervises confident nodes; LLM supervises
        # only nodes not already covered by GNN.
        gnn_only_mask = gnn_mask
        llm_only_mask = llm_mask & ~gnn_mask

        # Per-node sample weights (normalized inside loss)
        w_gnn = gnn_conf.clamp_min(0.0)
        w_llm = llm_outputs.get("confidences", torch.zeros_like(llm_pred).float()).to(self.device).float().clamp_min(0.0)

        loss_gt, loss_gt_prob, loss_gt_ce = self._teacher_supervision_loss(
            logits, gnn_probs, gnn_pred, gnn_only_mask,
            use_soft_prob=True,
            use_hard_ce=True,
            sample_weights=w_gnn,
        )
        loss_lt, loss_lt_prob, loss_lt_ce = self._teacher_supervision_loss(
            logits, llm_probs, llm_pred, llm_only_mask,
            use_soft_prob=True,
            use_hard_ce=True,
            sample_weights=w_llm,
        )

        # Node-count proportional scaling: each mean-reduced loss is scaled by
        # its share of total supervised nodes so that per-node gradient influence
        # is balanced across teacher groups (prevents small LLM set from
        # dominating the larger GNN set).
        n_gnn = gnn_only_mask.float().sum()
        n_llm = llm_only_mask.float().sum()
        n_total = (n_gnn + n_llm).clamp_min(1.0)
        s_gnn = n_gnn / n_total
        s_llm = n_llm / n_total

        any_supervised = llm_mask.any() or gnn_mask.any()
        if any_supervised:
            total = self.alpha * s_gnn * loss_gt + (1.0 - self.alpha) * s_llm * loss_lt
        else:
            total = logits.sum() * 0.0
        total.backward()
        self.student_optimizer.step()

        with torch.no_grad():
            total_mask = llm_mask | gnn_mask
            if total_mask.any():
                sem_conf = float(F.softmax(logits[total_mask], dim=-1).max(dim=-1).values.mean().item())
            else:
                sem_conf = 0.0

        return {
            "total": float(total.item()),
            "loss_gt": float(loss_gt.item()),
            "loss_gt_prob": float(loss_gt_prob.item()),
            "loss_gt_ce": float(loss_gt_ce.item()),
            "loss_lt": float(loss_lt.item()),
            "loss_lt_prob": float(loss_lt_prob.item()),
            "loss_lt_ce": float(loss_lt_ce.item()),
            "semantic_confidence": sem_conf,
            "llm_kept": int(llm_mask.sum().item()),
            "gnn_kept": int(gnn_mask.sum().item()),
            "scale_gnn": float(s_gnn.item()),
            "scale_llm": float(s_llm.item()),
        }

    def evaluate(self, data) -> Dict[str, float]:
        self.student.eval()
        graph = data.to(self.device)
        logits = self.student(graph)["logits"]
        predictions = logits.argmax(dim=-1)
        labels = data.y.to(self.device)

        metrics: Dict[str, float] = {}
        for split_name in ("query_mask", "train_mask", "val_mask", "test_mask"):
            if not hasattr(data, split_name):
                continue
            mask = getattr(data, split_name).to(self.device).bool()
            if int(mask.sum().item()) == 0:
                metrics[f"{split_name}_acc"] = 0.0
                continue
            metrics[f"{split_name}_acc"] = float((predictions[mask] == labels[mask]).float().mean().item())
        return metrics

    def run(
        self,
        data,
        dataset_name: str,
        label_names,
        output_dir: str,
        query_ratio: float,
        k_shot: int,
        seed: int,
        gnn_teacher_epochs: int,
        student_epochs: int,
    ) -> Dict[str, float]:
        if self.teacher_mode != "llm_only":
            gnn_cache_path = self._gnn_teacher_cache_path(dataset_name=dataset_name, seed=seed, data=data)
            cache_loaded = False
            if os.path.exists(gnn_cache_path):
                try:
                    payload = torch.load(gnn_cache_path, map_location="cpu")
                    cached_emb = payload.get("fixed_structural_map")
                    cached_head = payload.get("gnn_outputs")
                    if isinstance(cached_emb, torch.Tensor) and isinstance(cached_head, dict):
                        fixed_structural_map = cached_emb.to(self.device)
                        gnn_outputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in cached_head.items()}
                        cache_loaded = True
                        print(f"[GNN Teacher] Loaded cached outputs: {gnn_cache_path}")
                except Exception as exc:
                    print(f"[GNN Teacher] Cache load failed, rebuilding ({exc}).")

            if not cache_loaded:
                self.gnn_teacher.fit(data, epochs=gnn_teacher_epochs, verbose=True)
                gnn_teacher_ssl = self.gnn_teacher.infer(data)
                fixed_structural_map = gnn_teacher_ssl.embeddings.detach().to(self.device)
                gnn_outputs = self._train_structural_logit_head(
                    fixed_structural_map=fixed_structural_map,
                    data=data,
                    seed=seed,
                )
                try:
                    torch.save(
                        {
                            "fixed_structural_map": fixed_structural_map.detach().cpu(),
                            "gnn_outputs": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in gnn_outputs.items()},
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "ssl_method": str(self.gnn_teacher.ssl_method),
                        },
                        gnn_cache_path,
                    )
                    print(f"[GNN Teacher] Saved cache: {gnn_cache_path}")
                except Exception as exc:
                    print(f"[GNN Teacher] Cache save skipped ({exc}).")

            # Re-select query nodes from GNN-uncertain pool if enabled
            if self.llm_query_from_gnn_uncertain_pool and self.teacher_mode != "gnn_only":
                gnn_conf = gnn_outputs["confidences"].detach().cpu()
                train_mask = data.train_mask.cpu().bool()
                k_shot_mask = getattr(data, "k_shot_mask", torch.zeros_like(train_mask)).cpu().bool()
                candidates_mask = train_mask & ~k_shot_mask
                gnn_threshold = self._resolve_gnn_confidence_threshold(
                    gnn_conf.to(self.device),
                    candidates_mask.to(self.device),
                )
                # Restrict query pool to nodes below GNN confidence threshold.
                uncertain_mask = candidates_mask & (gnn_conf < float(gnn_threshold))
                uncertain_idx = torch.where(uncertain_mask)[0]
                uncertain_confs = gnn_conf[uncertain_idx]
                # Sort ascending: most uncertain first
                sorted_order = torch.argsort(uncertain_confs)
                # Query budget is defined as a ratio of the uncertain pool.
                ratio = float(query_ratio)
                if ratio < 0.0 or ratio > 1.0:
                    raise ValueError("query_ratio must be in [0, 1].")
                query_size = int(uncertain_idx.numel() * ratio)
                if ratio > 0.0 and query_size == 0 and int(uncertain_idx.numel()) > 0:
                    query_size = 1
                new_query_indices = uncertain_idx[sorted_order[:query_size]]
                new_query_mask = torch.zeros_like(data.query_mask)
                new_query_mask[new_query_indices] = True
                data.query_mask = new_query_mask
                data.unqueried_mask = train_mask & ~new_query_mask
                if query_size > 0:
                    conf_slice = uncertain_confs[sorted_order[:query_size]]
                    print(
                        f"[GNN-Uncertain] Re-selected {query_size}/{int(uncertain_idx.numel())} query nodes "
                        f"below threshold {float(gnn_threshold):.3f} "
                        f"({self.gnn_confidence_threshold_mode}"
                        f"{' q=' + str(self.gnn_confidence_quantile) if self.gnn_confidence_threshold_mode == 'quantile' else ''}) "
                        f"(GNN conf range: {conf_slice.min():.3f}~{conf_slice.max():.3f})"
                    )
                else:
                    print(
                        f"[GNN-Uncertain] Re-selected 0 query nodes "
                        f"(no nodes below threshold {float(gnn_threshold):.3f})."
                    )
        else:
            print("[GNN Teacher] Skipped (teacher_mode=llm_only)")
            gnn_outputs = self._empty_gnn_outputs(data)

        if self.teacher_mode != "gnn_only":
            repo_root = Path(__file__).resolve().parents[2]
            default_cache_dir = str((repo_root / "datasets" / "llm_cache").resolve())
            llm_teacher = DualTeacherLLM(
                dataset_name=dataset_name,
                label_names=label_names,
                model_name=self.llm_model_name,
                temperature=self.llm_temperature,
                max_output_tokens=self.llm_max_output_tokens,
                cache_dir=os.environ.get("LLM_CACHE_DIR", default_cache_dir),
                embedder_device=self.llm_embedder_device,
                cache_seed=seed,
                enable_queryset_cache=False,
            )
            llm_outputs = llm_teacher.query(data, output_path=None, k_shot=int(k_shot), write_logs=False)
        else:
            print("[LLM Teacher] Skipped (teacher_mode=gnn_only)")
            llm_outputs = self._empty_llm_outputs(data)

        self._resolve_gnn_confidence_threshold(
            gnn_outputs["confidences"].to(self.device).float(),
            data.train_mask.to(self.device).bool(),
        )
        teacher_diagnostics = self._compute_teacher_diagnostics(
            data=data,
            llm_outputs=llm_outputs,
            gnn_outputs=gnn_outputs,
        )
        # Query-level GNN confidence is currently hard-zeroed; suppress noisy log.

        history = []
        for epoch in range(student_epochs):
            stats = self._student_epoch(data, llm_outputs, gnn_outputs)
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(
                    f"[Student] Epoch {epoch + 1}/{student_epochs} | "
                    f"Total={stats['total']:.4f} | "
                    f"GT={stats['loss_gt']:.4f} | "
                    f"LT={stats['loss_lt']:.4f} | "
                    f"GTprob={stats['loss_gt_prob']:.4f} | "
                    f"LTprob={stats['loss_lt_prob']:.4f} | "
                    f"SemConf={stats['semantic_confidence']:.3f} | "
                    f"LLMKept={stats['llm_kept']} | "
                    f"GNNKept={stats['gnn_kept']}"
                )
            history.append(stats)

        llm_mask, gnn_mask = self._teacher_masks(data, llm_outputs, gnn_outputs)
        metrics = self.evaluate(data)
        metrics["llm_valid_parsed_query_count"] = int(llm_outputs["valid_query_mask"].sum().item())
        metrics["llm_kept_supervision_count"] = int(llm_mask.sum().item())
        metrics["gnn_kept_supervision_count"] = int(gnn_mask.sum().item())
        metrics["teacher_mode"] = self.teacher_mode
        metrics.update(teacher_diagnostics)

        return metrics
