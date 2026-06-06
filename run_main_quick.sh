#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cora}"          # ex) cora|citeseer|pubmed|wikics
BACKBONE="${2:-gcn}"      # ex) gcn|gat|graphsage
K_SHOT="${3:-3}"                # ex) 1|3|5|7|10
QR="${4:-0.8}"                  # query_ratio
ALPHA="${5:-0.6}"               # alpha
GNN_THR="${6:-0.5}"            # gnn threshold
LLM_THR="${7:-0.5}"            # llm threshold

export LLM_CACHE_STRICT_PROMPT_KEY="${LLM_CACHE_STRICT_PROMPT_KEY:-0}"

python src/main_dual_teacher_ssl.py \
  --dataset "$DATASET" \
  --teacher_mode dual \
  --student_backbone "$BACKBONE" \
  --k_shot "$K_SHOT" \
  --query_ratio "$QR" \
  --alpha "$ALPHA" \
  --gnn_confidence_mode top1_prob \
  --gnn_confidence_threshold "$GNN_THR" \
  --gnn_confidence_threshold_mode quantile \
  --gnn_confidence_quantile "$GNN_THR" \
  --llm_confidence_threshold "$LLM_THR" \
  --llm_model "${LLM_MODEL:-gpt-3.5-turbo}"
