"""Real embedder and reranker (sentence-transformers, CPU). Install with: uv sync --extra models

Defaults (both multilingual, both handle Hindi):
  embedder  intfloat/multilingual-e5-small  118M params, MIT; needs "query: "/"passage: " prefixes
  reranker  BAAI/bge-reranker-v2-m3         568M params, Apache-2.0
"""

import numpy as np

EMBEDDER = "intfloat/multilingual-e5-small"
RERANKER = "BAAI/bge-reranker-v2-m3"


class E5Embedder:
    def __init__(self, name: str = EMBEDDER):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device="cpu")

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self.model.encode(
            texts, prompt=prefix, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")


class CrossEncoderReranker:
    def __init__(self, name: str = RERANKER):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.model.predict([(query, t) for t in texts], batch_size=16).tolist()
