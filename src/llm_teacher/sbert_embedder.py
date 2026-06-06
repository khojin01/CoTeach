"""Text embedding utilities for the dual-teacher pipeline."""

from typing import Dict, List, Optional, Tuple, Union

import torch


SUPPORTED_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
}


class SBERTEmbedder:
    """Sentence-transformers embedder restricted to supported public models."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cuda",
        embedding_dimensions: Optional[int] = None,
    ):
        if embedding_dimensions not in (None, 0):
            raise ValueError("embedding_dimensions is not used for sentence-transformers backends.")

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. "
                f"Supported: {sorted(SUPPORTED_EMBEDDING_MODELS)}"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from exc

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

        self.reset_usage_stats()


    def reset_usage_stats(self) -> None:
        self._usage_stats = {
            "backend": "sbert",
            "model": self.model_name,
            "requests": 0,
            "input_count": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

    def get_usage_stats(self) -> Dict[str, object]:
        return dict(getattr(self, "_usage_stats", {
            "backend": "sbert",
            "model": self.model_name,
            "requests": 0,
            "input_count": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }))

    def _normalize_texts(self, texts: Union[str, List[str]]) -> Tuple[List[str], bool]:
        if isinstance(texts, str):
            return [texts], True
        return list(texts), False

    @staticmethod
    def _sanitize_texts(texts: List[str]) -> List[str]:
        return [t if isinstance(t, str) and t.strip() else "No reasoning provided" for t in texts]

    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        normalized, single = self._normalize_texts(texts)
        valid_texts = self._sanitize_texts(normalized)

        self._usage_stats["requests"] += 1
        self._usage_stats["input_count"] += len(valid_texts)
        embeddings = self.model.encode(
            valid_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
        ).to(dtype=torch.float32)

        if single:
            return embeddings[0]
        return embeddings

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        valid_texts = self._sanitize_texts(texts)
        self._usage_stats["requests"] += 1
        self._usage_stats["input_count"] += len(valid_texts)
        return self.model.encode(
            valid_texts,
            convert_to_tensor=True,
            show_progress_bar=len(texts) > 100,
            batch_size=batch_size,
            device=self.device,
        ).to(dtype=torch.float32)


_global_embedders: Dict[Tuple[str, str, Optional[int]], SBERTEmbedder] = {}


def get_embedder(
    model_name: str = "all-mpnet-base-v2",
    device: str = "cuda",
    embedding_dimensions: Optional[int] = None,
) -> SBERTEmbedder:
    key = (model_name, device, embedding_dimensions)
    if key not in _global_embedders:
        _global_embedders[key] = SBERTEmbedder(
            model_name=model_name,
            device=device,
            embedding_dimensions=embedding_dimensions,
        )
    return _global_embedders[key]


def embed_reasoning(reasoning_texts: Union[str, List[str]], device: str = "cuda") -> torch.Tensor:
    embedder = get_embedder(device=device)
    return embedder.embed(reasoning_texts)
