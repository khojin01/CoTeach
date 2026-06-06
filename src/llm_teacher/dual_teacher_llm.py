"""LLM-teacher querying, parsing, and pseudo-label logging."""

from __future__ import annotations

import json
import glob
import os
import re
import time
import hashlib
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml

from llm_prompts import get_dataset_metadata


def _load_openai_api_key(config_path: str = "config.yaml") -> str:
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    alt_env_key = os.getenv("OPENAI_KEY", "")
    if alt_env_key:
        return alt_env_key

    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        Path(config_path),
        Path.cwd() / config_path,
        repo_root / config_path,
        repo_root / "config.yaml",
    ]

    seen = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with open(resolved, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            api_key = str(config.get("OPENAI_KEY", "") or "")
            if api_key:
                return api_key
        except Exception:
            continue
    return ""


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


CACHE_LAYOUT_VERSION = 2


@dataclass
class ParsedLLMResponse:
    label: Optional[str]
    confidence: float
    probs: Optional[list]
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.label is not None and self.confidence >= 0.0


class DualTeacherLLM:
    """Text-only LLM teacher for pseudo-label generation."""

    def __init__(
        self,
        dataset_name: str,
        label_names: Sequence[str],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        cache_dir: str = "datasets/llm_cache",
        api_key: Optional[str] = None,
        embedder_device: str = "cuda",
        cache_seed: Optional[int] = None,
        enable_queryset_cache: bool = False,
    ):
        self.dataset_name = dataset_name.lower()
        self.label_names = [str(label) for label in label_names]
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.cache_dir = cache_dir
        self.api_key = api_key or _load_openai_api_key()
        self.embedder_device = embedder_device
        self.cache_seed = None if cache_seed is None else int(cache_seed)
        self.enable_queryset_cache = bool(enable_queryset_cache)
        self._client = None
        self._label_alias_map = self._build_label_alias_map(self.label_names)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_cache_dir = os.path.join(
            self.cache_dir,
            self.dataset_name,
            self.model_name.replace("/", "_"),
        )
        self.node_cache_dir = os.path.join(self.model_cache_dir, "nodes")
        self.migration_meta_path = os.path.join(self.model_cache_dir, "_migration_meta.json")
        os.makedirs(self.node_cache_dir, exist_ok=True)
        self.global_cache_path = os.path.join(
            self.cache_dir,
            f"{self.dataset_name}_{self.model_name.replace('/', '_')}_global_cache.json",
        )
        self._maybe_migrate_scripts_cache()

    def _maybe_migrate_scripts_cache(self) -> None:
        """Best-effort migration from `scripts/datasets/llm_cache` into the repo-root cache.

        If experiments were launched from inside `scripts/`, relative cache paths could resolve to
        `scripts/datasets/llm_cache`. This splits caches across two roots. When the configured cache dir
        is the repo-root `datasets/llm_cache`, we opportunistically copy missing *node-cache* files from
        the scripts cache into the root cache.
        """
        try:
            cache_dir_abs = os.path.abspath(self.cache_dir)
            if not cache_dir_abs.endswith(os.path.join("datasets", "llm_cache")):
                return

            repo_root = Path(__file__).resolve().parents[2]
            scripts_cache_root = repo_root / "scripts" / "datasets" / "llm_cache"
            if not scripts_cache_root.is_dir():
                return

            scripts_model_dir = scripts_cache_root / self.dataset_name / self.model_name.replace("/", "_")
            scripts_nodes_dir = scripts_model_dir / "nodes"
            if not scripts_nodes_dir.is_dir():
                return

            def copy_missing(src: Path, dst: Path) -> int:
                if not src.is_dir():
                    return 0
                dst.mkdir(parents=True, exist_ok=True)
                copied = 0
                for item in src.iterdir():
                    if not item.is_file():
                        continue
                    target = dst / item.name
                    if target.exists():
                        continue
                    shutil.copy2(str(item), str(target))
                    copied += 1
                return copied

            copied_nodes = copy_missing(scripts_nodes_dir, Path(self.node_cache_dir))
            if copied_nodes:
                print(f"[LLM] Migrated node caches from scripts/: nodes+{copied_nodes} -> {self.model_cache_dir}")
        except Exception as exc:
            print(f"[LLM] scripts cache migration skipped: {type(exc).__name__}: {exc}")

    def _is_cache_record_compatible(
        self,
        record: Dict[str, object],
        prompt_key: Optional[str] = None,
        require_prompt_key: bool = True,
    ) -> bool:
        if not isinstance(record, dict):
            return False
        response_text = str(record.get("response", "") or "").strip()
        if not response_text:
            return False
            return False
        if require_prompt_key and prompt_key is not None and str(record.get("prompt_key", "") or "") != str(prompt_key):
            return False
        return True

    def _parsed_from_cache_record(self, record: Optional[Dict[str, object]]) -> Optional[ParsedLLMResponse]:
        if not isinstance(record, dict):
            return None
        parsed_label = record.get("parsed_label")
        if parsed_label not in self.label_names:
            return None
        parsed_probs = record.get("parsed_probs")
        if parsed_probs is not None:
            try:
                parsed_probs = [float(value) for value in parsed_probs]
            except Exception:
                parsed_probs = None
            if parsed_probs is not None and len(parsed_probs) != len(self.label_names):
                parsed_probs = None
        # Always use top1(class_probs) as LLM confidence for thresholding.
        if parsed_probs is not None and len(parsed_probs) > 0:
            parsed_confidence = max(float(value) for value in parsed_probs)
        else:
            parsed_confidence = record.get("parsed_confidence")
        if parsed_confidence is None:
            parsed_confidence = 1.0
        try:
            parsed_confidence = float(parsed_confidence)
        except Exception:
            return None
        return ParsedLLMResponse(
            label=str(parsed_label),
            confidence=parsed_confidence,
            probs=parsed_probs,
            warning=record.get("warning"),
        )

    def _compute_margin_confidence(self, probs: Optional[List[float]]) -> Optional[float]:
        if probs is None or len(probs) == 0:
            return None
        try:
            sorted_probs = sorted(float(value) for value in probs)
        except Exception:
            return None
        if len(sorted_probs) == 1:
            return max(0.0, min(1.0, sorted_probs[-1]))
        margin = sorted_probs[-1] - sorted_probs[-2]
        return max(0.0, min(1.0, margin))

    def _extract_reported_confidence(self, record: Dict[str, object]) -> Optional[float]:
        if not isinstance(record, dict):
            return None
        if record.get("reported_confidence") is not None:
            try:
                return float(record.get("reported_confidence"))
            except Exception:
                return None
        response_text = str(record.get("response", "") or "").strip()
        if not response_text:
            return None
        try:
            payload = json.loads(response_text)
        except Exception:
            return None
        raw_conf = payload.get("confidence", None)
        try:
            return float(raw_conf)
        except Exception:
            return None

    def _write_output(self, output_path: str, output: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)

    def _load_existing_output(self, output_path: str) -> Dict[str, object]:
        if not output_path or not os.path.exists(output_path):
            return {}
        try:
            with open(output_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _load_global_cache(self) -> Dict[str, object]:
        if not os.path.exists(self.global_cache_path):
            return {"entries": {}, "node_entries": {}}
        try:
            with open(self.global_cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return {"entries": {}, "node_entries": {}}
            payload.setdefault("entries", {})
            payload.setdefault("node_entries", {})
            return payload
        except Exception:
            return {"entries": {}, "node_entries": {}}

    def _write_global_cache(self, cache_payload: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.global_cache_path), exist_ok=True)
        with open(self.global_cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache_payload, handle, indent=2, ensure_ascii=False)

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
        os.makedirs(self.node_cache_dir, exist_ok=True)
        payload = dict(record)
        payload["node_id"] = int(node_id)
        path = self._node_cache_path(node_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _prompt_cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _node_cache_key(self, node_id: int) -> str:
        return f"{self.dataset_name}:node:{int(node_id)}"

    def _query_set_cache_path(self, query_count: int, k_shot: int = 0) -> str:
        seed_suffix = f"_seed{self.cache_seed}" if self.cache_seed is not None else ""
        kshot_suffix = f"_kshot{int(k_shot)}"
        filename = (
            f"{self.dataset_name}_{self.model_name.replace('/', '_')}_"
            f"count{int(query_count)}{kshot_suffix}{seed_suffix}_queryset_cache.json"
        )
        return os.path.join(self.model_cache_dir, "querysets", filename)

    def _load_query_set_cache(self, query_count: int, k_shot: int = 0) -> Dict[str, object]:
        if not self.enable_queryset_cache:
            return {}
        cache_path = self._query_set_cache_path(query_count, k_shot=k_shot)
        candidate_paths = [cache_path]
        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return {}

    def _write_query_set_cache(self, query_count: int, payload: Dict[str, object], k_shot: int = 0) -> None:
        if not self.enable_queryset_cache:
            return
        cache_path = self._query_set_cache_path(query_count, k_shot=k_shot)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _restore_from_cached_queries(
        self,
        cached_queries: List[Dict[str, object]],
        data,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        llm_probs: torch.Tensor,
        valid_prob_mask: torch.Tensor,
        llm_logits: torch.Tensor,
        valid_logit_mask: torch.Tensor,
        llm_one_hot: torch.Tensor,
        valid_query_mask: torch.Tensor,
    ) -> Dict[str, object]:
        for record in cached_queries:
            node_id = int(record.get("node_id", -1))
            if node_id < 0 or node_id >= int(data.num_nodes):
                continue

            if not bool(record.get("valid_parse", False)):
                continue

            parsed_label = record.get("parsed_label")
            if parsed_label not in self.label_names:
                continue

            label_idx = self.label_names.index(parsed_label)
            predictions[node_id] = label_idx
            cached_confidence = record.get("parsed_confidence")
            confidences[node_id] = float(cached_confidence) if cached_confidence is not None else 1.0
            llm_one_hot[node_id, label_idx] = 1.0
            valid_query_mask[node_id] = True
            cached_probs = record.get("parsed_probs")
            if isinstance(cached_probs, list) and len(cached_probs) == llm_probs.size(-1):
                probs_t = torch.tensor(cached_probs, dtype=llm_probs.dtype, device=llm_probs.device)
                probs_t = probs_t / probs_t.sum().clamp_min(1e-8)
                llm_probs[node_id] = probs_t
                valid_prob_mask[node_id] = True
                llm_logits[node_id] = torch.log(probs_t.clamp_min(1e-8))
                valid_logit_mask[node_id] = True
                continue

            default_logits = torch.tensor(
                self._default_logits_from_label(label_idx),
                dtype=llm_logits.dtype,
                device=llm_logits.device,
            )
            probs_t = F.softmax(default_logits, dim=-1)
            llm_probs[node_id] = probs_t
            valid_prob_mask[node_id] = True
            llm_logits[node_id] = torch.log(probs_t.clamp_min(1e-8))
            valid_logit_mask[node_id] = True

        return {
            "restored_valid_queries": int(valid_query_mask.sum().item()),
        }

    def _build_query_records_from_node_cache(
        self,
        query_indices: List[int],
        data,
        few_shot_examples: List[Dict[str, str]] = None,
    ) -> List[Dict[str, object]]:
        cached_queries: List[Dict[str, object]] = []
        for node_id in query_indices:
            record = self._load_node_cache(int(node_id))
            prompt_key = None
            if few_shot_examples is not None:
                prompt = self.build_prompt(
                    title=data.title_texts[node_id],
                    abstract=data.abstract_texts[node_id],
                    few_shot_examples=few_shot_examples,
                )
                prompt_key = self._prompt_cache_key(prompt)
            if not self._is_cache_record_compatible(
                record or {},
                prompt_key=prompt_key,
                require_prompt_key=(prompt_key is not None),
            ):
                continue
            cached_queries.append({
                "node_id": int(node_id),
                "prompt_key": record.get("prompt_key"),
                "title": record.get("title", data.title_texts[node_id]),
                "abstract": record.get("abstract", data.abstract_texts[node_id]),
                "ground_truth": record.get("ground_truth"),
                "prompt": record.get("prompt", ""),
                "response": record.get("response", ""),
                "parsed_label": record.get("parsed_label"),
                "parsed_confidence": record.get("parsed_confidence"),
                "parsed_probs": record.get("parsed_probs"),
                "valid_parse": bool(record.get("valid_parse", False)),
                "warning": record.get("warning"),
            })
        return cached_queries

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key is missing. Set `OPENAI_API_KEY` or `OPENAI_KEY` in config.yaml."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _build_label_alias_map(self, label_names: Sequence[str]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for canonical in label_names:
            canonical = str(canonical)
            alias_map[_normalize_label(canonical)] = canonical

            # Pattern: "ML (Machine Learning)" -> aliases: "ML", "Machine Learning"
            m = re.match(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$", canonical)
            if m:
                lhs = m.group(1).strip()
                rhs = m.group(2).strip()
                if lhs:
                    alias_map[_normalize_label(lhs)] = canonical
                if rhs:
                    alias_map[_normalize_label(rhs)] = canonical
            else:
                # Pattern: "cs.LG" -> alias "LG" (not mandatory but harmless)
                if "." in canonical:
                    tail = canonical.split(".")[-1].strip()
                    if tail:
                        alias_map[_normalize_label(tail)] = canonical
        return alias_map

    def _resolve_label_name(self, text: object) -> Optional[str]:
        norm = _normalize_label(str(text or ""))
        if not norm:
            return None
        return self._label_alias_map.get(norm)

    def build_prompt(
        self,
        title: str,
        abstract: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        metadata = get_dataset_metadata(self.dataset_name)
        category_lines = [f"- {label}" for label in self.label_names]
        class_probs_lines = []
        for idx, label in enumerate(self.label_names):
            comma = "," if idx < len(self.label_names) - 1 else ""
            class_probs_lines.append(f'    "{label}": 0.00{comma}')

        sections = [
            "[System]",
            f"You are a careful classifier for {metadata['domain']}.",
            "",
            "[Task]",
            f"Read the title and abstract and estimate a probability distribution over the {len(self.label_names)} candidate categories.",
            "Candidate Categories:",
            *category_lines,
            "",
            "[Target Paper]",
            f"Title: {title}",
            f"Abstract: {abstract}",
            "",
            "Requirements:",
            "- `predicted_label` must be exactly one candidate category.",
            "- `class_probs` must contain all candidate categories exactly once.",
            "- All probabilities must be decimals between 0.00 and 1.00.",
            "- The probabilities must sum to exactly 1.00.",
            "- For `predicted_label`, use the canonical category string exactly as listed (do not shorten or paraphrase labels).",
            "- Only assign a very high probability if the title and abstract contain strong and specific evidence for that category.",
            "- Do not output explanations, reasons, or extra text.",
            "",
            "Output format:",
            "{",
            '  "predicted_label": "<one category>",',
            '  "class_probs": {',
            *class_probs_lines,
            "  }",
            "}",
        ]
        return "\n".join(sections)

    def _parse_probabilities(self, class_probs: object) -> Optional[List[float]]:
        if not isinstance(class_probs, dict):
            return None

        # Accept canonical keys and alias keys (e.g., ML -> ML (Machine Learning)).
        values: List[float] = [0.0 for _ in self.label_names]
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

        if len(seen) != len(self.label_names):
            return None
        if any(value < 0.0 for value in values):
            return None
        total = sum(values)
        if total <= 0.0:
            return None
        probs = [value / total for value in values]
        return probs

    def _default_logits_from_label(self, label_idx: int, epsilon: float = 1e-3) -> List[float]:
        num_classes = len(self.label_names)
        off_prob = epsilon / max(1, num_classes - 1)
        on_prob = 1.0 - epsilon
        probs = [off_prob] * num_classes
        probs[label_idx] = on_prob
        return [float(torch.log(torch.tensor(max(prob, 1e-8))).item()) for prob in probs]

    def parse_response(self, response_text: str, logprobs_content: Optional[list] = None) -> ParsedLLMResponse:
        if not response_text or not response_text.strip():
            return ParsedLLMResponse(
                label=None,
                confidence=-1.0,
                probs=None,
                warning="empty_response",
            )

        json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not json_match:
            return ParsedLLMResponse(
                label=None,
                confidence=-1.0,
                probs=None,
                warning="parse_failed_json",
            )
        try:
            payload = json.loads(json_match.group(0))
        except Exception:
            return ParsedLLMResponse(
                label=None,
                confidence=-1.0,
                probs=None,
                warning="parse_failed_json",
            )

        predicted_label = str(payload.get("predicted_label", "") or "").strip()
        label = self._resolve_label_name(predicted_label)
        if label is None:
            return ParsedLLMResponse(
                label=None,
                confidence=-1.0,
                probs=None,
                warning="parse_failed_label",
            )

        probs = self._parse_probabilities(payload.get("class_probs"))
        if probs is None:
            return ParsedLLMResponse(
                label=None,
                confidence=-1.0,
                probs=None,
                warning="parse_failed_probs",
            )
        warning = None
        confidence = max(0.0, min(1.0, max(float(value) for value in probs)))
        max_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        if self.label_names[max_idx] != label:
            warning = "predicted_label_mismatch" if warning is None else f"{warning};predicted_label_mismatch"
            label = self.label_names[max_idx]
        if margin < 0.10 and confidence > 0.80:
            warning = "overconfident_ambiguous" if warning is None else f"{warning};overconfident_ambiguous"

        return ParsedLLMResponse(
            label=label,
            confidence=confidence,
            probs=probs,
            warning=warning,
        )

    @staticmethod
    def _usage_to_dict(usage) -> Dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(usage, "model_dump"):
            raw = usage.model_dump()
        elif isinstance(usage, dict):
            raw = usage
        else:
            raw = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        prompt_tokens = int(raw.get("prompt_tokens") or 0)
        completion_tokens = int(raw.get("completion_tokens") or 0)
        total_tokens = int(raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return "ratelimit" in text or "rate limit" in text or "429" in text

    def _call_openai(self, prompt: str) -> tuple[str, Optional[list], Dict[str, int]]:
        max_retries = int(os.getenv("OPENAI_RATE_LIMIT_RETRIES", "8"))
        sleep_seconds = int(os.getenv("OPENAI_RATE_LIMIT_SLEEP_SECONDS", "60"))
        for attempt in range(max_retries + 1):
            try:
                if not self.api_key:
                    raise RuntimeError("OPENAI_API_KEY not set (config.yaml or env).")
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                message = response.choices[0].message
                return (message.content or "").strip(), None, self._usage_to_dict(getattr(response, "usage", None))
            except Exception as exc:
                if self._is_rate_limit_error(exc) and attempt < max_retries:
                    print(
                        f"[LLM] Rate limit hit; sleeping {sleep_seconds}s "
                        f"before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(sleep_seconds)
                    continue
                print(f"[LLM] OpenAI call failed: {type(exc).__name__}: {exc}")
                return "", None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return "", None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _make_output_path(self, dataset_name: str, k_shot: int, query_ratio: float) -> str:
        filename = f"{dataset_name}_k{k_shot}_qr{query_ratio:.3f}_llm_queries.json"
        return os.path.join(self.cache_dir, filename)

    def _build_k_shot_examples(self, data, k_shot: int) -> List[Dict[str, str]]:
        if k_shot <= 0 or not hasattr(data, "k_shot_mask"):
            return []
        k_shot_mask = data.k_shot_mask.cpu().bool()
        if int(k_shot_mask.sum().item()) == 0:
            return []
        examples: List[Dict[str, str]] = []
        for class_idx, label_name in enumerate(self.label_names):
            class_nodes = torch.where(k_shot_mask & (data.y.cpu() == class_idx))[0].tolist()
            class_nodes = class_nodes[: int(k_shot)]
            for node_id in class_nodes:
                raw_text = ""
                if hasattr(data, "raw_texts") and node_id < len(data.raw_texts):
                    raw_text = str(data.raw_texts[node_id])
                if not raw_text:
                    title = data.title_texts[node_id] if hasattr(data, "title_texts") else ""
                    abstract = data.abstract_texts[node_id] if hasattr(data, "abstract_texts") else ""
                    raw_text = f"Title: {title}\nAbstract: {abstract}"
                examples.append({"text": raw_text, "label": label_name})
        return examples

    def query(self, data, output_path: Optional[str] = None, k_shot: int = 0, write_logs: bool = True) -> Dict[str, torch.Tensor]:
        """Query the selected nodes, drop parse failures, and emit JSON logs."""
        device = data.y.device
        num_nodes = int(data.num_nodes)
        num_classes = len(self.label_names)

        predictions = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        confidences = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        reasoning_embeddings = torch.zeros((num_nodes, 768), dtype=torch.float32, device=device)
        llm_one_hot = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
        valid_query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        llm_probs = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
        valid_prob_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        llm_logits = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
        valid_logit_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        if output_path is None and write_logs:
            base = max(1, int(getattr(data, "query_budget_base_size_effective", int(data.query_mask.sum().item()) or 1)))
            inferred_ratio = float(data.query_mask.sum().item()) / float(base)
            output_path = self._make_output_path(
                dataset_name=self.dataset_name,
                k_shot=int(k_shot),
                query_ratio=inferred_ratio,
            )

        query_indices = torch.where(data.query_mask.cpu())[0].tolist()
        total_requested = len(query_indices)
        few_shot_examples: List[Dict[str, str]] = self._build_k_shot_examples(data=data, k_shot=int(k_shot))
        node_cache_file_count = len(glob.glob(os.path.join(self.node_cache_dir, "*.json")))
        query_set_cache_path = None

        print(f"[LLM] Cache config: node_cache_dir={self.node_cache_dir} (files={node_cache_file_count})")

        logs: List[Dict[str, object]] = []
        warnings: List[Dict[str, object]] = []
        synthesized_query_set = self._build_query_records_from_node_cache(
            query_indices=query_indices,
            data=data,
            few_shot_examples=few_shot_examples,
        ) if query_indices else []
        if synthesized_query_set:
            print(
                f"[LLM] Synthesized query-set from node cache: "
                f"{len(synthesized_query_set)}/{total_requested} nodes."
            )
            self._restore_from_cached_queries(
                cached_queries=synthesized_query_set,
                data=data,
                predictions=predictions,
                confidences=confidences,
                llm_probs=llm_probs,
                valid_prob_mask=valid_prob_mask,
                llm_logits=llm_logits,
                valid_logit_mask=valid_logit_mask,
                llm_one_hot=llm_one_hot,
                valid_query_mask=valid_query_mask,
            )
            logs = list(synthesized_query_set)
            completed_node_ids = {
                int(record["node_id"])
                for record in synthesized_query_set
                if isinstance(record, dict) and int(record.get("node_id", -1)) >= 0
            }
        else:
            completed_node_ids = set()

        if completed_node_ids:
            synthesized_cache_payload = {
                "metadata": {
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "label_names": self.label_names,
                    "k_shot": int(k_shot),
                    "k_shot_examples": len(few_shot_examples),
                    "num_requested_queries": total_requested,
                    "num_completed_queries": len(completed_node_ids),
                    "num_valid_queries": int(valid_query_mask.sum().item()),
                    "cache_mode": "node_cache_synthesized_reuse",
                    "source_cache_path": None,
                    "node_cache_dir": self.node_cache_dir,
                },
                "queries": logs,
                "warnings": [],
                "summary": {
                    "accuracy_against_ground_truth": None,
                    "num_invalid_queries": total_requested - int(valid_query_mask.sum().item()),
                    "excluded_node_ids": [],
                },
            }
            if len(completed_node_ids) == total_requested and int(valid_query_mask.sum().item()) > 0:
                print(
                    "[LLM] All requested queries satisfied from cache "
                    f"({len(completed_node_ids)}/{total_requested}). Skipping live queries."
                )
                output = {
                    "metadata": dict(synthesized_cache_payload["metadata"]),
                    "queries": logs,
                    "warnings": [],
                    "summary": dict(synthesized_cache_payload["summary"]),
                }
                if write_logs and output_path:
                    self._write_output(output_path, output)
                return {
                    "predictions": predictions,
                    "confidences": confidences,
                    "reasoning_embeddings": reasoning_embeddings,
                    "valid_query_mask": valid_query_mask,
                    "llm_one_hot": llm_one_hot,
                    "probs": llm_probs,
                    "valid_prob_mask": valid_prob_mask,
                    "logits": llm_logits,
                    "valid_logit_mask": valid_logit_mask,
                    "log_path": output_path,
                }

        pending_query_indices = [node_id for node_id in query_indices if node_id not in completed_node_ids]
        if completed_node_ids:
            print(f"[LLM] Resuming from existing log: {len(completed_node_ids)}/{total_requested} queries already completed.")
        strict_prompt_key = str(os.getenv("LLM_CACHE_STRICT_PROMPT_KEY", "1")).strip().lower() not in {"0", "false", "no"}
        print(f"[LLM] Cache strict prompt-key match: {strict_prompt_key}")

        cache_plan = {
            "dataset_node_hits": 0,
            "prompt_hits": 0,
            "api_needed": 0,
        }
        planned_prompts: Dict[int, Dict[str, object]] = {}
        for node_id in pending_query_indices:
            prompt = self.build_prompt(
                title=data.title_texts[node_id],
                abstract=data.abstract_texts[node_id],
                few_shot_examples=few_shot_examples,
            )
            prompt_key = self._prompt_cache_key(prompt)
            node_key = self._node_cache_key(node_id)
            node_cached_entry = self._load_node_cache(node_id)
            node_cache_ok = self._is_cache_record_compatible(
                node_cached_entry or {},
                prompt_key=prompt_key,
                require_prompt_key=strict_prompt_key,
            )
            planned_prompts[node_id] = {
                "prompt": prompt,
                "prompt_key": prompt_key,
            }
            if node_cache_ok:
                cache_plan["dataset_node_hits"] += 1
            else:
                cache_plan["api_needed"] += 1

        if pending_query_indices:
            print(
                "[LLM] Cache plan: "
                f"query_set_done={len(completed_node_ids)} | "
                f"dataset_node_hits={cache_plan['dataset_node_hits']} | "
                f"prompt_hits={cache_plan['prompt_hits']} | "
                f"api_needed={cache_plan['api_needed']}"
            )

        runtime_cache_stats = {
            "dataset_node_hits": 0,
            "prompt_hits": 0,
            "api_called": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for offset, node_id in enumerate(pending_query_indices, start=1):
            prompt = str(planned_prompts.get(node_id, {}).get("prompt", ""))
            prompt_key = str(planned_prompts.get(node_id, {}).get("prompt_key", ""))
            if not prompt or not prompt_key:
                prompt = self.build_prompt(
                    title=data.title_texts[node_id],
                    abstract=data.abstract_texts[node_id],
                    few_shot_examples=few_shot_examples,
                )
                prompt_key = self._prompt_cache_key(prompt)
            node_key = self._node_cache_key(node_id)
            node_cached_entry = self._load_node_cache(node_id)

            response_text = ""
            cache_source = None
            parsed = None
            openai_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if self._is_cache_record_compatible(
                node_cached_entry or {},
                prompt_key=prompt_key,
                require_prompt_key=strict_prompt_key,
            ):
                print(
                    f"[LLM] Reusing dataset-node cache for node {offset}/{len(pending_query_indices)} "
                    f"(global {len(completed_node_ids) + offset}/{total_requested})..."
                )
                response_text = str(node_cached_entry.get("response", "") or "")
                cache_source = "node"
                parsed = self._parsed_from_cache_record(node_cached_entry)
                cached_usage = node_cached_entry.get("openai_usage") if isinstance(node_cached_entry, dict) else None
                openai_usage = self._usage_to_dict(cached_usage)
                runtime_cache_stats["dataset_node_hits"] += 1
            if not str(response_text).strip():
                if cache_source:
                    print(f"[LLM] Cached {cache_source} entry empty; re-querying node {offset}/{len(pending_query_indices)}...")
                print(
                    f"[LLM] Querying node {offset}/{len(pending_query_indices)} "
                    f"(global {len(completed_node_ids) + offset}/{total_requested})..."
                )
                response_text, _, openai_usage = self._call_openai(prompt)
                runtime_cache_stats["api_called"] += 1
                runtime_cache_stats["prompt_tokens"] += int(openai_usage.get("prompt_tokens", 0))
                runtime_cache_stats["completion_tokens"] += int(openai_usage.get("completion_tokens", 0))
                runtime_cache_stats["total_tokens"] += int(openai_usage.get("total_tokens", 0))
                parsed = None
            if offset % 10 == 0 or offset == len(pending_query_indices):
                print(
                    f"[LLM] Progress: {offset}/{len(pending_query_indices)} "
                    f"(global {len(completed_node_ids) + offset}/{total_requested}) | "
                    f"api_called={runtime_cache_stats['api_called']} | "
                    f"node_cache_hits={runtime_cache_stats['dataset_node_hits']}"
                )
            if parsed is None:
                if not str(response_text).strip():
                    parsed = ParsedLLMResponse(
                        label=None,
                        confidence=-1.0,
                        probs=None,
                        warning="empty_response",
                    )
                else:
                    parsed = self.parse_response(response_text, None)

            ground_truth_idx = int(data.y[node_id].item())
            ground_truth_label = self.label_names[ground_truth_idx]

            record = {
                "node_id": node_id,
                "prompt_key": prompt_key,
                "title": data.title_texts[node_id],
                "abstract": data.abstract_texts[node_id],
                "ground_truth": ground_truth_label,
                "prompt": prompt,
                "response": response_text,
                "parsed_label": parsed.label,
                "parsed_confidence": parsed.confidence if parsed.confidence >= 0 else None,
                "reported_confidence": self._extract_reported_confidence({"response": response_text}),
                "parsed_probs": parsed.probs,
                "valid_parse": parsed.is_valid,
                "warning": parsed.warning,
                "openai_usage": openai_usage,
                "api_call": bool(openai_usage.get("total_tokens", 0)),
            }
            logs.append(record)
            if str(response_text).strip():
                node_record = {
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "node_id": int(node_id),
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    "title": data.title_texts[node_id],
                    "abstract": data.abstract_texts[node_id],
                    "response": response_text,
                    "parsed_label": parsed.label,
                    "parsed_confidence": parsed.confidence if parsed.confidence >= 0 else None,
                    "reported_confidence": self._extract_reported_confidence({"response": response_text}),
                    "parsed_probs": parsed.probs,
                    "valid_parse": parsed.is_valid,
                    "warning": parsed.warning,
                    "openai_usage": openai_usage,
                    "updated_at": datetime.now().isoformat(),
                }
                self._write_node_cache(node_id, node_record)

            if not parsed.is_valid:
                warnings.append(
                    {
                        "node_id": node_id,
                        "warning": parsed.warning or "parse_failed",
                    }
                )
                partial_output = {
                    "metadata": {
                        "dataset": self.dataset_name,
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        "label_names": self.label_names,
                        "k_shot": int(k_shot),
                        "k_shot_examples": len(few_shot_examples),
                        "num_requested_queries": total_requested,
                        "num_completed_queries": len(completed_node_ids) + offset,
                        "num_valid_queries": int(valid_query_mask.sum().item()),
                        "node_cache_dir": self.node_cache_dir,
                        "token_usage": {
                            "prompt_tokens": int(runtime_cache_stats["prompt_tokens"]),
                            "completion_tokens": int(runtime_cache_stats["completion_tokens"]),
                            "total_tokens": int(runtime_cache_stats["total_tokens"]),
                            "api_called": int(runtime_cache_stats["api_called"]),
                        },
                    },
                    "queries": logs,
                    "warnings": warnings,
                    "summary": {
                        "accuracy_against_ground_truth": None,
                        "num_invalid_queries": len(warnings),
                        "excluded_node_ids": [warning["node_id"] for warning in warnings],
                    },
                }
                if write_logs and output_path:
                    self._write_output(output_path, partial_output)
                continue

            label_idx = self.label_names.index(parsed.label)
            predictions[node_id] = label_idx
            confidences[node_id] = parsed.confidence
            llm_one_hot[node_id, label_idx] = 1.0
            valid_query_mask[node_id] = True
            if parsed.probs is not None:
                probs_t = torch.tensor(parsed.probs, dtype=torch.float32, device=device)
                probs_t = probs_t / probs_t.sum().clamp_min(1e-8)
                llm_probs[node_id] = probs_t
                valid_prob_mask[node_id] = True
                llm_logits[node_id] = torch.log(probs_t.clamp_min(1e-8))
                valid_logit_mask[node_id] = True
            partial_output = {
                "metadata": {
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "label_names": self.label_names,
                    "k_shot": int(k_shot),
                    "k_shot_examples": len(few_shot_examples),
                    "num_requested_queries": total_requested,
                    "num_completed_queries": len(completed_node_ids) + offset,
                    "num_valid_queries": int(valid_query_mask.sum().item()),
                        "node_cache_dir": self.node_cache_dir,
                        "token_usage": {
                            "prompt_tokens": int(runtime_cache_stats["prompt_tokens"]),
                            "completion_tokens": int(runtime_cache_stats["completion_tokens"]),
                            "total_tokens": int(runtime_cache_stats["total_tokens"]),
                            "api_called": int(runtime_cache_stats["api_called"]),
                        },
                    },
                "queries": logs,
                "warnings": warnings,
                "summary": {
                    "accuracy_against_ground_truth": None,
                    "num_invalid_queries": len(warnings),
                    "excluded_node_ids": [warning["node_id"] for warning in warnings],
                },
            }
            if write_logs and output_path:
                self._write_output(output_path, partial_output)
            time.sleep(0.1)

        valid_indices = torch.where(valid_query_mask)[0]
        accuracy = 0.0
        if valid_indices.numel() > 0:
            accuracy = float((predictions[valid_indices] == data.y[valid_indices]).float().mean().item())

        output = {
            "metadata": {
                "dataset": self.dataset_name,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "label_names": self.label_names,
                "k_shot": int(k_shot),
                "k_shot_examples": len(few_shot_examples),
                "num_requested_queries": total_requested,
                "num_completed_queries": len(logs),
                "num_valid_queries": int(valid_query_mask.sum().item()),
                "node_cache_dir": self.node_cache_dir,
                "cache_stats": runtime_cache_stats,
                "token_usage": {
                    "prompt_tokens": int(runtime_cache_stats["prompt_tokens"]),
                    "completion_tokens": int(runtime_cache_stats["completion_tokens"]),
                    "total_tokens": int(runtime_cache_stats["total_tokens"]),
                    "api_called": int(runtime_cache_stats["api_called"]),
                },
            },
            "queries": logs,
            "warnings": warnings,
            "summary": {
                "accuracy_against_ground_truth": accuracy,
                "num_invalid_queries": total_requested - int(valid_query_mask.sum().item()),
                "excluded_node_ids": [warning["node_id"] for warning in warnings],
            },
        }
        if write_logs and output_path:
            self._write_output(output_path, output)

        print(
            "[LLM] Query run complete: "
            f"dataset_node_hits={runtime_cache_stats['dataset_node_hits']} | "
            f"prompt_hits={runtime_cache_stats['prompt_hits']} | "
            f"api_called={runtime_cache_stats['api_called']} | "
            f"valid_queries={int(valid_query_mask.sum().item())}/{total_requested}"
        )

        return {
            "predictions": predictions,
            "confidences": confidences,
            "reasoning_embeddings": reasoning_embeddings,
            "valid_query_mask": valid_query_mask,
            "llm_one_hot": llm_one_hot,
            "probs": llm_probs,
            "valid_prob_mask": valid_prob_mask,
            "logits": llm_logits,
            "valid_logit_mask": valid_logit_mask,
            "log_path": output_path,
        }
