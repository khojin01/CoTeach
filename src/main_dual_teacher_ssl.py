"""CLI entrypoint for the modular dual-teacher SSL framework."""

from __future__ import annotations

import argparse
import os
import sys

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.dual_teacher_data import apply_dual_teacher_split, load_dual_teacher_dataset
from pipeline.main_helpers import set_seed
from pipeline.dual_teacher_trainer import DualTeacherTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-teacher SSL framework")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed", "arxiv", "wikics"])
    parser.add_argument("--query_ratio", type=float, default=0.1)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query_selection_method", type=str, default="cluster_random", choices=["random", "cluster_random"])
    parser.add_argument("--query_selection_feature_source", type=str, default="structural", choices=["structural", "semantic"])
    parser.add_argument("--query_selection_num_clusters", type=int, default=0)

    parser.add_argument("--ssl_method", type=str, default="dgi", choices=["dgi", "bgrl", "graphcl", "gca"])
    parser.add_argument("--student_backbone", type=str, default="gcn", choices=["graphsage", "gcn", "gat"])
    parser.add_argument("--teacher_mode", type=str, default="dual", choices=["dual", "gnn_only", "llm_only"])

    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--gnn_teacher_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_teacher_embedding_dim", type=int, default=768)
    parser.add_argument("--student_hidden_dim", type=int, default=256)
    parser.add_argument("--student_alignment_dim", type=int, default=768)
    parser.add_argument("--gnn_teacher_epochs", type=int, default=300)
    parser.add_argument("--student_epochs", type=int, default=200)
    parser.add_argument("--gnn_teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_output_tokens", type=int, default=400)
    parser.add_argument(
        "--text_embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    )
    parser.add_argument("--struct_node2vec_dim", type=int, default=0)

    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--gnn_logit_epochs", type=int, default=200)
    parser.add_argument("--gnn_logit_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_logit_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--gnn_confidence_mode",
        type=str,
        default="top1_prob",
        choices=["top1_logit", "margin", "top1_prob", "margin_prob", "entropy"],
    )
    parser.add_argument("--gnn_confidence_threshold", type=float, default=float("-inf"))
    parser.add_argument(
        "--gnn_confidence_threshold_mode",
        type=str,
        default="absolute",
        choices=["absolute", "quantile"],
        help="Use a fixed numeric GNN threshold or compute it from the train-pool confidence quantile.",
    )
    parser.add_argument(
        "--gnn_confidence_quantile",
        type=float,
        default=0.5,
        help="Quantile used when --gnn_confidence_threshold_mode=quantile. Example: 0.5 keeps the top half.",
    )
    parser.add_argument("--llm_confidence_threshold", type=float, default=0.0)
    parser.set_defaults(
        gnn_priority_llm_fallback=True,
        llm_query_from_gnn_uncertain_pool=True,
    )
    parser.add_argument(
        "--gnn_priority_llm_fallback",
        dest="gnn_priority_llm_fallback",
        action="store_true",
        help="Use GNN supervision where confident; use LLM supervision only on remaining nodes (default: enabled).",
    )
    parser.add_argument(
        "--no_gnn_priority_llm_fallback",
        dest="gnn_priority_llm_fallback",
        action="store_false",
        help="Disable GNN-priority LLM fallback routing.",
    )
    parser.add_argument(
        "--llm_query_from_gnn_uncertain_pool",
        dest="llm_query_from_gnn_uncertain_pool",
        action="store_true",
        help="Re-select LLM query nodes from GNN-uncertain pool after GNN teacher training (default: enabled).",
    )
    parser.add_argument(
        "--no_llm_query_from_gnn_uncertain_pool",
        dest="llm_query_from_gnn_uncertain_pool",
        action="store_false",
        help="Disable re-selecting LLM query nodes from the GNN-uncertain pool.",
    )

    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got "
            f"{args.train_ratio}+{args.val_ratio}+{args.test_ratio}={ratio_sum}"
        )
    set_seed(args.seed)

    output_dir = args.output_dir or ""

    print("=" * 72)
    print("Dual-Teacher SSL Framework")
    print("=" * 72)
    print(f"Dataset          : {args.dataset}")
    print(f"Split ratio      : {args.train_ratio}:{args.val_ratio}:{args.test_ratio}")
    print(f"K-shot / Query   : {args.k_shot} / {args.query_ratio}")
    print(f"Query Selection  : {args.query_selection_method} ({args.query_selection_feature_source}, K={args.query_selection_num_clusters or 'auto'})")
    print(f"SSL / Student    : {args.ssl_method} / {args.student_backbone}")
    print(f"Teacher Mode     : {args.teacher_mode}")
    print("=" * 72)

    data, canonical_name = load_dual_teacher_dataset(
        dataset_name=args.dataset,
        device=args.device,
        sbert_model_name=args.text_embedding_model,
        struct_node2vec_dim=args.struct_node2vec_dim,
    )
    data = apply_dual_teacher_split(
        data=data,
        dataset_name=canonical_name,
        query_ratio=args.query_ratio,
        k_shot=args.k_shot,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        query_selection_method=args.query_selection_method,
        query_selection_feature_source=args.query_selection_feature_source,
        query_selection_num_clusters=args.query_selection_num_clusters,
        teacher_mode=args.teacher_mode,
    )

    print(
        f"[Split] train={int(data.train_mask.sum().item())}, "
        f"k-shot={int(getattr(data, 'k_shot_mask', torch.zeros_like(data.train_mask)).sum().item())}, "
        f"query={int(data.query_mask.sum().item())}, "
        f"unqueried={int(data.unqueried_mask.sum().item())}, "
        f"val={int(data.val_mask.sum().item())}, "
        f"test={int(data.test_mask.sum().item())}"
    )

    trainer = DualTeacherTrainer(
        semantic_input_dim=int(data.x.size(-1)),
        structural_input_dim=int(data.structural_x.size(-1)),
        num_classes=len(data.label_names),
        ssl_method=args.ssl_method,
        student_backbone=args.student_backbone,
        alpha=args.alpha,
        gnn_teacher_hidden_dim=args.gnn_teacher_hidden_dim,
        gnn_teacher_embedding_dim=args.gnn_teacher_embedding_dim,
        student_hidden_dim=args.student_hidden_dim,
        student_alignment_dim=args.student_alignment_dim,
        gnn_teacher_lr=args.gnn_teacher_lr,
        student_lr=args.student_lr,
        weight_decay=args.weight_decay,
        llm_model_name=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_output_tokens=args.llm_max_output_tokens,
        device=args.device,
        distill_temperature=args.distill_temperature,
        gnn_logit_epochs=args.gnn_logit_epochs,
        gnn_logit_lr=args.gnn_logit_lr,
        gnn_logit_hidden_dim=args.gnn_logit_hidden_dim,
        gnn_confidence_mode=args.gnn_confidence_mode,
        gnn_confidence_threshold=args.gnn_confidence_threshold,
        gnn_confidence_threshold_mode=args.gnn_confidence_threshold_mode,
        gnn_confidence_quantile=args.gnn_confidence_quantile,
        llm_confidence_threshold=args.llm_confidence_threshold,
        gnn_priority_llm_fallback=args.gnn_priority_llm_fallback,
        llm_query_from_gnn_uncertain_pool=args.llm_query_from_gnn_uncertain_pool,
        teacher_mode=args.teacher_mode,
    )

    metrics = trainer.run(
        data=data,
        dataset_name=canonical_name,
        label_names=data.label_names,
        output_dir=output_dir,
        query_ratio=args.query_ratio,
        k_shot=args.k_shot,
        seed=args.seed,
        gnn_teacher_epochs=args.gnn_teacher_epochs,
        student_epochs=args.student_epochs,
    )

    print(f"val acc: {metrics.get('val_mask_acc', 0.0):.4f}  | test acc: {metrics.get('test_mask_acc', 0.0):.4f}")


if __name__ == "__main__":
    main()
