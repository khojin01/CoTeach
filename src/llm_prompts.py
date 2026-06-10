"""LLM teacher prompt/response utilities for a single JSON-distribution prompt."""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Prompt strategy tag (also used in cache directory names)
LLM_PROMPT_STRATEGY = "json_class_probs_v1"


# Dataset metadata
DATASET_METADATA = {
    'pubmed': {
        'object_type': 'Paper',
        'domain': 'medical research papers',
        'label_names': ['Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 'Experimental'],
        'question': 'Which category does this paper belong to?',
        'description': 'diabetes research'
    },
    'cora': {
        'object_type': 'Paper',
        'domain': 'computer science papers',
        'label_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                        'Probabilistic_Methods', 'Reinforcement_Learning', 
                        'Rule_Learning', 'Theory'],
        'question': 'Which category does this paper belong to?',
        'description': 'machine learning papers'
    },
    'citeseer': {
        'object_type': 'Paper',
        'domain': 'computer science papers',
        'label_names': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
        'question': 'Which category does this paper belong to?',
        'description': 'computer science papers'
    },
    'arxiv': {
        'object_type': 'Paper',
        'domain': 'computer science papers',
        'label_names': [
            'cs.NA', 'cs.MM', 'cs.LO', 'cs.CY', 'cs.CR', 'cs.DC', 'cs.HC', 'cs.CE',
            'cs.NI', 'cs.CC', 'cs.AI', 'cs.MA', 'cs.GL', 'cs.NE', 'cs.SC', 'cs.AR',
            'cs.CV', 'cs.GR', 'cs.ET', 'cs.SY', 'cs.CG', 'cs.OH', 'cs.PL', 'cs.SE',
            'cs.LG', 'cs.SD', 'cs.SI', 'cs.RO', 'cs.IT', 'cs.PF', 'cs.CL', 'cs.IR',
            'cs.MS', 'cs.FL', 'cs.DS', 'cs.OS', 'cs.GT', 'cs.DB', 'cs.DL', 'cs.DM'
        ],
        'question': 'Which arXiv CS sub-category does this paper belong to?',
        'description': 'Computer Science papers'
    },
    'wikics': {
        'object_type': 'Wiki Article',
        'domain': 'wiki articles',
        'label_names': [
            'Computational linguistics',
            'Databases',
            'Operating systems',
            'Computer architecture',
            'Computer security',
            'Internet protocols',
            'Computer file systems',
            'Distributed computing architecture',
            'Web technology',
            'Programming language topics',
        ],
        'question': 'Which wiki category does this article belong to?',
        'description': 'computer science concepts'
    }
}

def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Return dataset metadata."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_METADATA:
        supported = sorted(list(DATASET_METADATA.keys()))
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {supported}")
    return DATASET_METADATA[dataset_name]


def create_zero_shot_top3_prompt(
    node_features: Optional[Any],
    dataset_name: str,
    label_names: Optional[List[str]] = None,
    node_text: Optional[str] = None,
    k_shot_examples: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> str:
    """Build the single supported JSON-distribution prompt format."""
    metadata = get_dataset_metadata(dataset_name)

    if label_names is None:
        label_names = metadata['label_names']

    raw_text = (node_text or "").strip()
    title = "Untitled"
    abstract = "No abstract available."
    if raw_text:
        if "Title:" in raw_text and "Abstract:" in raw_text:
            title_match = re.search(r"Title:\s*(.+?)(?:\n|Abstract:)", raw_text, re.DOTALL)
            abstract_match = re.search(r"Abstract:\s*(.+)$", raw_text, re.DOTALL)
            if title_match:
                title = title_match.group(1).strip() or title
            if abstract_match:
                abstract = abstract_match.group(1).strip() or abstract
        else:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if len(lines) >= 2:
                title = lines[0]
                abstract = " ".join(lines[1:])
            elif lines:
                title = lines[0]
                abstract = lines[0]

    category_lines = [f"- {label}" for label in label_names]
    class_probs_lines = []
    for idx, label in enumerate(label_names):
        suffix = "," if idx < len(label_names) - 1 else ""
        class_probs_lines.append(f'    "{label}": 0.00{suffix}')

    sections = [
        "[System]",
        f"You are a careful classifier for {metadata['domain']}.",
        "[Task]",
        f"Read the title and abstract and estimate a probability distribution over the {len(label_names)} candidate categories.",
        "Candidate Categories:",
        *category_lines,
        "[Target Paper]",
        f"Title: {title}",
        f"Abstract: {abstract}",
        "Requirements:",
        "- `predicted_label` must be exactly one candidate category.",
        "- `class_probs` must contain all candidate categories exactly once.",
        "- All probabilities must be decimals between 0.00 and 1.00.",
        "- The probabilities must sum to exactly 1.00.",
        "- For `predicted_label`, use the canonical category string exactly as listed (do not shorten or paraphrase labels).",
        "- Only assign a very high probability if the title and abstract contain strong and specific evidence for that category.",
        "- Do not output explanations, reasons, or extra text.",
        "Output format:",
        "{",
        '  "predicted_label": "<one category>",',
        '  "class_probs": {',
        *class_probs_lines,
        "  }",
        "}",
    ]
    return "\n".join(sections)


def create_prompt(node_features=None, dataset_name=None, label_names=None, 
                  node_text=None, k_shot_examples=None, **kwargs):
    """Compatibility wrapper for the single supported prompt format."""
    return create_zero_shot_top3_prompt(
        node_features=node_features,
        dataset_name=dataset_name,
        label_names=label_names,
        node_text=node_text,
        k_shot_examples=k_shot_examples,
        **kwargs
    )


def _normalize_label_text(text: Any) -> str:
    """Normalize label text for robust matching."""
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[\s_\-/:]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]", "", normalized)
    return normalized


def _fuzzy_match_label(predicted: str, candidate: str) -> bool:
    """Fuzzy match between predicted label and candidate label"""
    if not predicted or not candidate:
        return False
    
    pred_norm = _normalize_label_text(predicted)
    cand_norm = _normalize_label_text(candidate)
    
    # Exact match
    if pred_norm == cand_norm:
        return True
    
    # Contains match
    if pred_norm in cand_norm or cand_norm in pred_norm:
        return True
    
    return False


def _normalize_confidence(confidence: Any) -> float:
    """Normalize confidence into [0, 1]."""
    try:
        if isinstance(confidence, str):
            confidence = confidence.replace("%", "").strip()
        conf = float(confidence)
        if conf > 1.0:
            conf = conf / 100.0
        return float(max(0.0, min(1.0, conf)))
    except Exception:
        return 0.0


def _resolve_label(answer: Any, label_names: List[str], response_text: str = "") -> Optional[str]:
    """Map heterogeneous answer formats to canonical labels."""
    if not label_names:
        return None

    # 1) Handle integer-like outputs (e.g., 2, "2", "class 2")
    idx_match = re.search(r"(\d+)", str(answer)) if answer is not None else None
    if isinstance(answer, int) or (isinstance(answer, str) and answer.strip().isdigit()):
        idx = int(answer)
        if 0 <= idx < len(label_names):
            return label_names[idx]
    elif idx_match and any(k in str(answer).lower() for k in ["class", "label", "category"]):
        idx = int(idx_match.group(1))
        if 0 <= idx < len(label_names):
            return label_names[idx]

    # 2) Direct / normalized matching
    answer_str = "" if answer is None else str(answer).strip()
    if answer_str in label_names:
        return answer_str

    normalized_map = {_normalize_label_text(label): label for label in label_names}
    answer_norm = _normalize_label_text(answer_str)
    if answer_norm in normalized_map:
        return normalized_map[answer_norm]

    # 3) Try matching labels from full response text
    if response_text:
        response_norm = _normalize_label_text(response_text)
        matched = [label for key, label in normalized_map.items() if key and key in response_norm]
        if len(matched) == 1:
            return matched[0]

    # 4) Fuzzy fallback
    if answer_str:
        from difflib import get_close_matches
        matches = get_close_matches(answer_norm, list(normalized_map.keys()), n=1, cutoff=0.7)
        if matches:
            return normalized_map[matches[0]]

    return None


def _extract_class_probabilities(prob_obj: Any, label_names: List[str]) -> Optional[List[float]]:
    """Convert class probabilities into label_names order, normalized to sum to 1."""
    if prob_obj is None or not label_names:
        return None

    probs = None

    if isinstance(prob_obj, dict):
        vec = np.zeros(len(label_names), dtype=float)
        assigned = np.zeros(len(label_names), dtype=bool)

        for k, v in prob_obj.items():
            try:
                p = float(v)
            except Exception:
                continue

            # Label string key
            resolved = _resolve_label(k, label_names)
            if resolved is not None:
                idx = label_names.index(resolved)
                vec[idx] = p
                assigned[idx] = True
                continue

            # Numeric key
            try:
                idx = int(k)
                if 0 <= idx < len(label_names):
                    vec[idx] = p
                    assigned[idx] = True
            except Exception:
                continue

        if assigned.any():
            probs = vec

    elif isinstance(prob_obj, list):
        values = []
        for v in prob_obj:
            try:
                values.append(float(v))
            except Exception:
                values.append(0.0)
        if len(values) == len(label_names):
            probs = np.array(values, dtype=float)

    if probs is None:
        return None

    probs = np.clip(probs, a_min=0.0, a_max=None)
    s = float(probs.sum())
    if s <= 0:
        return None
    probs = probs / s
    return probs.tolist()


def parse_llm_response(response_text: str, label_names: List[str]) -> Dict[str, Any]:
    """Parse a JSON-style label distribution response with legacy fallback."""
    if not response_text or not label_names:
        return {
            'answer': None,
            'reasoning': ''
        }

    result = {'answer': None, 'reasoning': ''}

    try:
        payload = json.loads(response_text)
        predicted = payload.get("predicted_label")
        resolved = _resolve_label(predicted, label_names, response_text=response_text)
        if resolved is not None:
            result["answer"] = resolved
            return result
    except Exception:
        pass

    category_match = re.search(r'Final Category:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if category_match:
        predicted_category = category_match.group(1).strip()
        for label in label_names:
            if _fuzzy_match_label(predicted_category, label):
                result['answer'] = label
                break
    return result
