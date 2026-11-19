"""LLM teacher of CoTeach.

For every node selected for querying, the LLM receives the node text (title + abstract) and the
candidate label names and returns a probability distribution over the labels as JSON. The
pseudo-label is the label with the highest probability and that probability is the confidence
score; the trainer later keeps only predictions whose confidence exceeds the LLM threshold.

Responses are cached per node under `<cache_dir>/<dataset>/<model>/nodes/<node_id>.json`, so a
node is sent to the API at most once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import torch

from llm_api_config import OpenAIChat, usage_to_dict
from llm_prompts import build_llm_prompt


def _normalize_label(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").strip().lower())


@dataclass
class ParsedLLMResponse:
    label: Optional[str]
    confidence: float
    probs: Optional[List[float]]
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.label is not None and self.probs is not None

    @classmethod
    def failed(cls, warning: str) -> "ParsedLLMResponse":
        return cls(label=None, confidence=-1.0, probs=None, warning=warning)


class DualTeacherLLM:
    """Text-only LLM teacher that produces pseudo-labels for the queried nodes."""

    def __init__(
        self,
        dataset_name: str,
        label_names: Sequence[str],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        cache_dir: str = "datasets/llm_cache",
        api_key: Optional[str] = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.label_names = [str(label) for label in label_names]
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._chat = OpenAIChat(model_name, temperature, max_output_tokens, api_key=api_key)
        self._label_alias_map = self._build_label_alias_map(self.label_names)
        self.node_cache_dir = os.path.join(cache_dir, self.dataset_name, model_name.replace("/", "_"), "nodes")
        os.makedirs(self.node_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ prompt
    def build_prompt(self, title: str, abstract: str) -> str:
        return build_llm_prompt(self.dataset_name, self.label_names, title, abstract)

    @staticmethod
    def _prompt_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------------- parsing
    @staticmethod
    def _build_label_alias_map(label_names: Sequence[str]) -> Dict[str, str]:
        """Map normalized names and simple aliases ("ML (Machine Learning)" -> "ML", "cs.LG" -> "LG")."""
        alias_map: Dict[str, str] = {}
        for canonical in label_names:
            alias_map[_normalize_label(canonical)] = canonical
            m = re.match(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$", canonical)
            aliases = [m.group(1), m.group(2)] if m else ([canonical.split(".")[-1]] if "." in canonical else [])
            for alias in aliases:
                if alias.strip():
                    alias_map[_normalize_label(alias)] = canonical
        return alias_map

    def _resolve_label_name(self, text: object) -> Optional[str]:
        norm = _normalize_label(text)
        return self._label_alias_map.get(norm) if norm else None

    def _parse_probabilities(self, class_probs: object) -> Optional[List[float]]:
        """Return the normalized distribution in label order, or None unless every label appears once."""
        if not isinstance(class_probs, dict):
            return None
        values = [0.0] * len(self.label_names)
        seen = set()
        for raw_key, raw_value in class_probs.items():
            canonical = self._resolve_label_name(raw_key)
            if canonical is None:
                continue
            idx = self.label_names.index(canonical)
            if idx in seen:
                continue
            try:
                values[idx] = float(raw_value)
            except Exception:
                return None
            seen.add(idx)
        total = sum(values)
        if len(seen) != len(self.label_names) or min(values) < 0.0 or total <= 0.0:
            return None
        return [value / total for value in values]

    def _from_probs(self, probs: List[float], stated_label: Optional[str]) -> ParsedLLMResponse:
        """Pseudo-label = argmax of the distribution (ties go to the LLM's stated label), confidence = its probability."""
        top = max(range(len(probs)), key=lambda idx: probs[idx])
        if stated_label is not None and probs[self.label_names.index(stated_label)] == probs[top]:
            top = self.label_names.index(stated_label)
        label = self.label_names[top]
        warning = "predicted_label_mismatch" if stated_label is not None and stated_label != label else None
        return ParsedLLMResponse(label=label, confidence=max(0.0, min(1.0, probs[top])), probs=probs, warning=warning)

    def parse_response(self, response_text: str) -> ParsedLLMResponse:
        if not response_text or not response_text.strip():
            return ParsedLLMResponse.failed("empty_response")
        json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        try:
            payload = json.loads(json_match.group(0)) if json_match else None
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return ParsedLLMResponse.failed("parse_failed_json")
        stated_label = self._resolve_label_name(str(payload.get("predicted_label", "") or "").strip())
        if stated_label is None:
            return ParsedLLMResponse.failed("parse_failed_label")
        probs = self._parse_probabilities(payload.get("class_probs"))
        if probs is None:
            return ParsedLLMResponse.failed("parse_failed_probs")
        return self._from_probs(probs, stated_label)

    # ------------------------------------------------------------------- cache
    def _node_cache_path(self, node_id: int) -> str:
        return os.path.join(self.node_cache_dir, f"{int(node_id)}.json")

    def _load_node_cache(self, node_id: int) -> Optional[Dict[str, object]]:
        path = self._node_cache_path(node_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _write_node_cache(self, node_id: int, record: Dict[str, object]) -> None:
        with open(self._node_cache_path(node_id), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)

    def _parse_cached(self, record: Dict[str, object]) -> ParsedLLMResponse:
        """Use the stored distribution when it is complete; otherwise re-parse the stored response."""
        if record.get("parsed_label") in self.label_names:
            try:
                stored = [float(value) for value in record.get("parsed_probs") or []]
            except Exception:
                stored = []
            if len(stored) == len(self.label_names) and min(stored) >= 0.0 and sum(stored) > 0.0:
                total = sum(stored)
                return self._from_probs([value / total for value in stored], str(record["parsed_label"]))
        return self.parse_response(str(record.get("response", "") or ""))

    # ------------------------------------------------------------------- query
    def query(self, data, output_path: Optional[str] = None, k_shot: int = 0, write_logs: bool = True) -> Dict[str, torch.Tensor]:
        """Query every node in `data.query_mask` (cache first, then API) and return node-level tensors.

        Unparseable responses are dropped (`valid_query_mask` stays False for those nodes).
        With LLM_CACHE_STRICT_PROMPT_KEY=0, a cached response is reused even if it was produced
        by a different prompt; with the default (1) the stored prompt hash must match.
        """
        device = data.y.device
        num_nodes, num_classes = int(data.num_nodes), len(self.label_names)
        predictions = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        confidences = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        probs_out = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
        valid_query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        strict = os.getenv("LLM_CACHE_STRICT_PROMPT_KEY", "1").strip().lower() not in {"0", "false", "no"}
        query_indices = torch.where(data.query_mask.cpu())[0].tolist()
        stats = {"cache_hits": 0, "api_called": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        logs: List[Dict[str, object]] = []
        print(f"[LLM] {len(query_indices)} query nodes | cache={self.node_cache_dir} | strict_prompt_key={strict}")

        for offset, node_id in enumerate(query_indices, start=1):
            title, abstract = data.title_texts[node_id], data.abstract_texts[node_id]
            prompt = self.build_prompt(title=title, abstract=abstract)
            prompt_key = self._prompt_key(prompt)

            cached = self._load_node_cache(node_id) or {}
            cache_hit = bool(str(cached.get("response", "") or "").strip()) and (
                not strict or str(cached.get("prompt_key", "") or "") == prompt_key
            )
            if cache_hit:
                stats["cache_hits"] += 1
                response_text = str(cached["response"])
                usage = usage_to_dict(cached.get("openai_usage"))
                parsed = self._parse_cached(cached)
            else:
                response_text, usage = self._chat(prompt)
                stats["api_called"] += 1
                for key in usage:
                    stats[key] += usage[key]
                parsed = self.parse_response(response_text)
                if response_text:
                    self._write_node_cache(node_id, {
                        "dataset": self.dataset_name,
                        "model": self.model_name,
                        "node_id": int(node_id),
                        "prompt_key": prompt_key,
                        "prompt": prompt,
                        "title": title,
                        "abstract": abstract,
                        "response": response_text,
                        "parsed_label": parsed.label,
                        "parsed_confidence": parsed.confidence if parsed.is_valid else None,
                        "parsed_probs": parsed.probs,
                        "valid_parse": parsed.is_valid,
                        "warning": parsed.warning,
                        "openai_usage": usage,
                        "updated_at": datetime.now().isoformat(),
                    })
                time.sleep(0.1)

            if parsed.is_valid:
                label_idx = self.label_names.index(parsed.label)
                predictions[node_id] = label_idx
                confidences[node_id] = parsed.confidence
                probs_t = torch.tensor(parsed.probs, dtype=torch.float32, device=device)
                probs_out[node_id] = probs_t / probs_t.sum().clamp_min(1e-8)
                valid_query_mask[node_id] = True

            if write_logs:
                logs.append({
                    "node_id": int(node_id),
                    "ground_truth": self.label_names[int(data.y[node_id].item())],  # log only, never used for training
                    "response": response_text,
                    "parsed_label": parsed.label,
                    "parsed_confidence": parsed.confidence if parsed.is_valid else None,
                    "parsed_probs": parsed.probs,
                    "valid_parse": parsed.is_valid,
                    "warning": parsed.warning,
                    "cache_hit": cache_hit,
                })
            if offset % 100 == 0 or offset == len(query_indices):
                print(f"[LLM] Progress {offset}/{len(query_indices)} | cache_hits={stats['cache_hits']} | api_called={stats['api_called']}")

        num_valid = int(valid_query_mask.sum().item())
        print(f"[LLM] Query run complete: valid={num_valid}/{len(query_indices)} | {stats}")

        if write_logs:
            if output_path is None:
                output_path = os.path.join(self.cache_dir, f"{self.dataset_name}_k{int(k_shot)}_llm_queries.json")
            valid_idx = torch.where(valid_query_mask)[0]
            accuracy = float((predictions[valid_idx] == data.y[valid_idx]).float().mean().item()) if num_valid else 0.0
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "metadata": {
                        "dataset": self.dataset_name,
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        "label_names": self.label_names,
                        "k_shot": int(k_shot),
                        "num_requested_queries": len(query_indices),
                        "num_valid_queries": num_valid,
                        "node_cache_dir": self.node_cache_dir,
                        "stats": stats,
                    },
                    "queries": logs,
                    "summary": {
                        "accuracy_against_ground_truth": accuracy,
                        "excluded_node_ids": [r["node_id"] for r in logs if not r["valid_parse"]],
                    },
                }, handle, indent=2, ensure_ascii=False)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "valid_query_mask": valid_query_mask,
            "probs": probs_out,
            "valid_prob_mask": valid_query_mask.clone(),
            "log_path": output_path if write_logs else None,
        }
