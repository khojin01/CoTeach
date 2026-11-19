"""OpenAI access for the LLM teacher: API key lookup and a chat-completion call with rate-limit retries."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def load_openai_api_key(config_path: str = "config.yaml") -> str:
    """Read the key from OPENAI_API_KEY / OPENAI_KEY, else from `OPENAI_KEY` in config.yaml."""
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        if os.getenv(name):
            return os.getenv(name, "")
    repo_root = Path(__file__).resolve().parents[1]
    for path in dict.fromkeys(p.resolve() for p in (Path(config_path), repo_root / "config.yaml")):
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                key = str((yaml.safe_load(handle) or {}).get("OPENAI_KEY", "") or "")
        except Exception:
            continue
        if key:
            return key
    return ""


def usage_to_dict(usage) -> Dict[str, int]:
    if usage is None:
        return dict(EMPTY_USAGE)
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {key: getattr(usage, key, 0) for key in EMPTY_USAGE}
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total_tokens = int(raw.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


class OpenAIChat:
    """Lazily created OpenAI client with one-prompt chat completion."""

    def __init__(self, model_name: str, temperature: float, max_output_tokens: int, api_key: Optional[str] = None):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.api_key = api_key or load_openai_api_key()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key is missing. Set `OPENAI_API_KEY` or `OPENAI_KEY` in config.yaml.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def __call__(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Return (response text, token usage); ("", zero usage) if the call fails."""
        max_retries = int(os.getenv("OPENAI_RATE_LIMIT_RETRIES", "8"))
        sleep_seconds = int(os.getenv("OPENAI_RATE_LIMIT_SLEEP_SECONDS", "60"))
        for attempt in range(max_retries + 1):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                content = (response.choices[0].message.content or "").strip()
                return content, usage_to_dict(getattr(response, "usage", None))
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}".lower()
                if ("ratelimit" in text or "rate limit" in text or "429" in text) and attempt < max_retries:
                    print(f"[LLM] Rate limit hit; sleeping {sleep_seconds}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(sleep_seconds)
                    continue
                print(f"[LLM] OpenAI call failed: {type(exc).__name__}: {exc}")
                break
        return "", dict(EMPTY_USAGE)
