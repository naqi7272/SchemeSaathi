import numpy as np
import pytest

from saathi.retrieval.search import BM25Index, DenseIndex, HybridRetriever, rrf
from saathi.retrieval.tokenize import tokenize
from tests.conftest import make_passage


def test_tokenize_keeps_devanagari_words_whole():
    assert tokenize("किसान सम्मान निधि") == ["किसान", "सम्मान", "निधि"]


def test_tokenize_strips_danda_and_lowercases():
    assert tokenize("लाभ ₹6000 है। PM-KISAN") == ["लाभ", "6000", "है", "pm", "kisan"]


def test_tokenize_hinglish():
    assert tokenize("Kitna paisa milta hai?") == ["kitna", "paisa", "milta", "hai"]


def passage(scheme, i, text):
    p = make_passage(scheme, "d0-x", i)
    return p.model_copy(update={"text": text})


CORPUS = [
    passage("kisan", 0, "Farmer families with cultivable land get income support."),
    passage("kisan", 1, "किसान परिवारों को हर साल ₹6000 की सहायता मिलती है।"),
    passage("pension", 0, "Pension for workers in the unorganised sector aged 18 to 40."),
    passage("scholar", 0, "Scholarship for students from SC and ST categories."),
]


def test_bm25_finds_english_and_hindi_matches():
    index = BM25Index(CORPUS)
    assert index.search("pension unorganised workers", k=2)[0] == "pension:d0-x:0"
    assert index.search("किसान सहायता", k=2)[0] == "kisan:d0-x:1"


def test_bm25_returns_nothing_for_no_overlap():
    assert BM25Index(CORPUS).search("zzz qqq", k=3) == []
    assert BM25Index(CORPUS).search("???", k=3) == []


class KeywordEmbedder:
    """Fake embedder: one dimension per keyword, so similarity is predictable."""

    WORDS = ["farmer", "pension", "scholarship", "किसान"]

    def _vec(self, texts):
        rows = [[float(w in t.lower()) for w in self.WORDS] + [0.01] for t in texts]
        m = np.array(rows, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    embed_queries = _vec
    embed_passages = _vec


def test_dense_index_ranks_by_cosine(tmp_path):
    index = DenseIndex(CORPUS, KeywordEmbedder(), path=tmp_path / "chroma")
    assert index.search("scholarship please", k=1) == ["scholar:d0-x:0"]
    assert len(index.search("anything", k=10)) == len(CORPUS)  # k capped at collection size


def test_dense_index_rebuild_replaces_collection(tmp_path):
    DenseIndex(CORPUS, KeywordEmbedder(), path=tmp_path)
    index = DenseIndex(CORPUS[:2], KeywordEmbedder(), path=tmp_path)
    assert index.collection.count() == 2


def test_rrf_rewards_agreement():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"
    assert rrf([["x"], []]) == ["x"]


def test_rrf_scores_exact():
    # b: 1/(60+2) + 1/(60+1); a: 1/(60+1); c: 1/(60+2)
    assert rrf([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


class ReverseReranker:
    """Fake reranker that prefers shorter passages."""

    def score(self, query, texts):
        return [-len(t) for t in texts]


def test_hybrid_retrieve_with_and_without_reranker():
    plain = HybridRetriever(CORPUS, KeywordEmbedder())
    top = plain.retrieve("farmer income support", k=2)
    assert top[0].passage_id == "kisan:d0-x:0"
    reranked = HybridRetriever(CORPUS, KeywordEmbedder(), reranker=ReverseReranker())
    got = reranked.retrieve("farmer", k=4)
    assert [len(p.text) for p in got] == sorted(len(p.text) for p in got)


@pytest.mark.parametrize("k", [1, 3])
def test_hybrid_respects_k(k):
    assert len(HybridRetriever(CORPUS, KeywordEmbedder()).retrieve("pension farmer", k=k)) == k
