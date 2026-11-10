"""Hybrid retrieval: BM25 + dense (ChromaDB), fused with reciprocal rank fusion, then an
optional cross-encoder reranker.

Embedders and rerankers are passed in, so tests use small fakes and the real
sentence-transformers models are only loaded by `saathi.retrieval.models`.
"""

from pathlib import Path
from typing import Protocol

import bm25s
import chromadb
import numpy as np

from saathi.retrieval.tokenize import tokenize
from saathi.schemas import Passage


class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class BM25Index:
    def __init__(self, passages: list[Passage]):
        self.ids = [p.passage_id for p in passages]
        self.model = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        self.model.index([tokenize(p.text) for p in passages], show_progress=False)

    def search(self, query: str, k: int) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in order if scores[i] > 0]


class DenseIndex:
    COLLECTION = "passages"

    def __init__(self, passages: list[Passage], embedder: Embedder, path: Path | None = None):
        self.embedder = embedder
        settings = chromadb.Settings(anonymized_telemetry=False)
        client = (
            chromadb.PersistentClient(path=str(path), settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        if self.COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(self.COLLECTION)
        self.collection = client.create_collection(
            self.COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
        vectors = embedder.embed_passages([p.text for p in passages])
        batch = 1000
        for start in range(0, len(passages), batch):
            chunk = passages[start : start + batch]
            self.collection.add(
                ids=[p.passage_id for p in chunk],
                embeddings=vectors[start : start + batch],
                metadatas=[{"scheme_id": p.scheme_id} for p in chunk],
            )

    def search(self, query: str, k: int) -> list[str]:
        vector = self.embedder.embed_queries([query])
        k = min(k, self.collection.count())
        result = self.collection.query(query_embeddings=vector, n_results=k)
        return result["ids"][0]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1.
    Ties keep the order in which documents were first seen."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


class HybridRetriever:
    def __init__(
        self,
        passages: list[Passage],
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
        chroma_path: Path | None = None,
    ):
        self.by_id = {p.passage_id: p for p in passages}
        self.bm25 = BM25Index(passages)
        self.dense = DenseIndex(passages, embedder, chroma_path)
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        fused = rrf(
            [self.bm25.search(query, self.candidates), self.dense.search(query, self.candidates)]
        )[: self.candidates]
        if self.reranker and fused:
            scores = self.reranker.score(query, [self.by_id[i].text for i in fused])
            fused = [i for _, i in sorted(zip(scores, fused, strict=True), key=lambda t: -t[0])]
        return [self.by_id[i] for i in fused[:k]]
