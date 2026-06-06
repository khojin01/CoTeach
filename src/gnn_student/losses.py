from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralAlignmentLoss(nn.Module):
    """Cosine similarity alignment between student projected embeddings and GNN teacher embeddings.
    
    Uses cosine similarity instead of MSE to focus on direction rather than magnitude,
    which is more suitable for normalized embeddings from SSL models like BGRL.
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_proj: torch.Tensor, teacher_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.bool().to(student_proj.device)
            student_proj = student_proj[mask]
            teacher_emb = teacher_emb[mask]
        if student_proj.numel() == 0:
            return torch.tensor(0.0, device=student_proj.device)
        
        # Normalize both embeddings to unit sphere
        student_norm = F.normalize(student_proj, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
        
        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
        
        # Loss = 1 - cosine_similarity (minimize to maximize similarity)
        loss = 1.0 - cosine_sim.mean()
        return loss


class SemanticClassificationLoss(nn.Module):
    """Cross-entropy on LLM pseudo-labels for queried nodes only."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        mask = mask.bool().to(logits.device)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        per_node = F.cross_entropy(logits[mask], pseudo_labels[mask].long(), reduction="none")
        if sample_weights is not None:
            weights = sample_weights.to(logits.device)[mask]
            weights = weights / weights.mean().clamp_min(1e-8)
            return (per_node * weights).mean()
        return per_node.mean()


class PrototypeDistillationLoss(nn.Module):
    """Class-aware prototype supervision in the teacher embedding space."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_proj: torch.Tensor,
        prototypes: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor,
        prototype_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        if student_proj.numel() == 0:
            return torch.tensor(0.0, device=student_proj.device)

        device = student_proj.device
        mask = mask.bool().to(device)
        if not mask.any():
            return torch.tensor(0.0, device=device)

        targets = pseudo_labels.to(device).long()
        valid_mask = mask.clone()

        if prototype_mask is not None:
            prototype_mask = prototype_mask.bool().to(device)
            valid_mask = valid_mask & prototype_mask[targets]
        else:
            prototype_mask = torch.ones(prototypes.size(0), dtype=torch.bool, device=device)

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        student_valid = F.normalize(student_proj[valid_mask], p=2, dim=-1)
        proto_valid = F.normalize(prototypes.to(device), p=2, dim=-1)
        logits = torch.matmul(student_valid, proto_valid.t()) / self.temperature
        logits[:, ~prototype_mask] = -1e9

        target_valid = targets[valid_mask]
        per_node = F.cross_entropy(logits, target_valid, reduction="none")

        if sample_weights is not None:
            weights = sample_weights.to(device)[valid_mask]
            weights = weights / weights.mean().clamp_min(1e-8)
            return (per_node * weights).mean()
        return per_node.mean()


class SoftLogitDistillationLoss(nn.Module):
    """KL distillation loss between student logits and teacher logits."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        if student_logits.numel() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        mask = mask.bool().to(student_logits.device)
        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device)

        temp = self.temperature
        student_log_probs = F.log_softmax(student_logits[mask] / temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits[mask] / temp, dim=-1)
        per_node = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        if sample_weights is not None:
            weights = sample_weights.to(student_logits.device)[mask]
            weights = weights / weights.mean().clamp_min(1e-8)
            per_node = per_node * weights
        return per_node.mean() * (temp ** 2)


class SoftProbDistillationLoss(nn.Module):
    """KL distillation loss between student logits and a teacher probability distribution.

    This is useful when the teacher provides normalized class probabilities directly
    (e.g., LLM `class_probs`) instead of logits/logprobs.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        if student_logits.numel() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        mask = mask.bool().to(student_logits.device)
        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device)

        temp = self.temperature
        student_log_probs = F.log_softmax(student_logits[mask] / temp, dim=-1)
        probs = teacher_probs[mask].to(student_logits.device)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs.clamp_min(1e-8)
        per_node = F.kl_div(student_log_probs, probs, reduction="none").sum(dim=-1)
        if sample_weights is not None:
            weights = sample_weights.to(student_logits.device)[mask]
            weights = weights / weights.mean().clamp_min(1e-8)
            per_node = per_node * weights
        return per_node.mean() * (temp ** 2)


# Compatibility losses for zero-shot modules.
class LLMDistillationLoss(nn.Module):
    """Masked CE on LLM pseudo-labels (zero-shot safe)."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, llm_predictions, mask, llm_confidences=None):
        device = student_logits.device
        if mask is None or not mask.any():
            return torch.tensor(0.0, device=device)
        logits_valid = student_logits[mask]
        targets_valid = llm_predictions[mask]
        if llm_confidences is not None:
            weights = llm_confidences[mask]
            weights = weights / (weights.mean() + 1e-8)
            per_node = F.cross_entropy(logits_valid, targets_valid, reduction="none")
            return (per_node * weights).mean()
        return F.cross_entropy(logits_valid, targets_valid)


class FeatureAlignmentLoss(nn.Module):
    """MSE alignment between student and target embeddings."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, student_emb, target_emb, mask=None, llm_confidences=None):
        device = student_emb.device
        if mask is not None:
            mask = mask.bool().to(device)
            student_emb = student_emb[mask]
            target_emb = target_emb[mask]
            if llm_confidences is not None:
                llm_confidences = llm_confidences[mask]
        if student_emb.numel() == 0:
            return torch.tensor(0.0, device=device)
        mse = self.mse(student_emb, target_emb).sum(dim=-1)
        if llm_confidences is not None:
            weights = llm_confidences / (llm_confidences.mean() + 1e-8)
            return (mse * weights).mean()
        return mse.mean()


class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emb, teacher_emb, mask=None):
        device = student_emb.device
        if mask is not None:
            mask = mask.bool().to(device)
            student_emb = student_emb[mask]
            teacher_emb = teacher_emb[mask]
        if student_emb.numel() == 0:
            return torch.tensor(0.0, device=device)
        student_emb = F.normalize(student_emb, p=2, dim=1)
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        sim_matrix = torch.matmul(student_emb, teacher_emb.t()) / self.temperature
        labels = torch.arange(student_emb.size(0), dtype=torch.long, device=device)
        return F.cross_entropy(sim_matrix, labels)
