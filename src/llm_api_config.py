"""Minimal LLM API configuration helpers."""

import os
from typing import Any, Dict

import yaml


class LLMConfig:
    """Load OpenAI configuration from config.yaml and environment variables."""

    def __init__(self):
        self.openai_api_key = ""
        self.default_model = "gpt-3.5-turbo"
        self.temperature = 0.0
        self.max_tokens = 500
        self.rate_limit_delay = 0.2

        self._load_config_from_yaml()
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY", "")

    def _load_config_from_yaml(self) -> None:
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            self.openai_api_key = str(config.get("OPENAI_KEY", "") or config.get("OPENAI_API_KEY", "")).strip()
        except Exception:
            self.openai_api_key = ""

    def get_openai_client(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install openai: pip install openai") from exc

        return OpenAI(api_key=self.openai_api_key)


def call_real_llm_api(prompt: str, provider: str = "openai") -> Dict[str, Any]:
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}. Only 'openai' is supported.")
    return call_openai_api(LLMConfig(), prompt)


def call_openai_api(config: LLMConfig, prompt: str) -> Dict[str, Any]:
    client = config.get_openai_client()
    try:
        response = client.chat.completions.create(
            model=config.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return {"content": response.choices[0].message.content}
    except Exception:
        return {"content": ""}
