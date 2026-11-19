"""Dataset metadata and the LLM-teacher prompt.

The prompt gives the node text (title + abstract) and the candidate label names only, and asks
for a JSON probability distribution over the labels.
"""

from typing import Any, Dict, Sequence

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "cora": {
        "domain": "computer science papers",
        "label_names": [
            "Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
            "Reinforcement_Learning", "Rule_Learning", "Theory",
        ],
    },
    "citeseer": {
        "domain": "computer science papers",
        "label_names": ["Agents", "AI", "DB", "IR", "ML", "HCI"],
    },
    "pubmed": {
        "domain": "medical research papers",
        "label_names": ["Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2", "Experimental"],
    },
    "wikics": {
        "domain": "wiki articles",
        "label_names": [
            "Computational linguistics", "Databases", "Operating systems", "Computer architecture",
            "Computer security", "Internet protocols", "Computer file systems",
            "Distributed computing architecture", "Web technology", "Programming language topics",
        ],
    },
    "arxiv": {
        "domain": "computer science papers",
        "label_names": [
            "cs.NA", "cs.MM", "cs.LO", "cs.CY", "cs.CR", "cs.DC", "cs.HC", "cs.CE",
            "cs.NI", "cs.CC", "cs.AI", "cs.MA", "cs.GL", "cs.NE", "cs.SC", "cs.AR",
            "cs.CV", "cs.GR", "cs.ET", "cs.SY", "cs.CG", "cs.OH", "cs.PL", "cs.SE",
            "cs.LG", "cs.SD", "cs.SI", "cs.RO", "cs.IT", "cs.PF", "cs.CL", "cs.IR",
            "cs.MS", "cs.FL", "cs.DS", "cs.OS", "cs.GT", "cs.DB", "cs.DL", "cs.DM",
        ],
    },
}


def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    name = dataset_name.lower()
    if name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset: {name}. Supported: {sorted(DATASET_METADATA)}")
    return DATASET_METADATA[name]


def build_llm_prompt(dataset_name: str, label_names: Sequence[str], title: str, abstract: str) -> str:
    """Prompt for one node. Changing the text changes the cache key of every node."""
    domain = get_dataset_metadata(dataset_name)["domain"]
    n = len(label_names)
    class_probs_lines = [f'    "{label}": 0.00{"," if idx < n - 1 else ""}' for idx, label in enumerate(label_names)]
    return "\n".join([
        "[System]",
        f"You are a careful classifier for {domain}.",
        "[Task]",
        f"Read the title and abstract and estimate a probability distribution over the {n} candidate categories.",
        "Candidate Categories:",
        *[f"- {label}" for label in label_names],
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
    ])
