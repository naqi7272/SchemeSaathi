import json

from saathi.arms.arms import ClosedBookArm, RagArm, parse_answer
from saathi.arms.prompts import SYSTEM_CLOSED_BOOK, SYSTEM_RAG, rag_messages
from saathi.llm.base import LLMResult, Usage
from tests.conftest import make_passage


class EchoModel:
    """Returns a fixed reply and remembers what it was sent."""

    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.seen: list[tuple[str, list]] = []

    def chat(self, system, messages, schema=None, max_tokens=0):
        self.seen.append((system, messages))
        usage = Usage(
            requested_model="fake",
            served_model="fake",
            input_tokens=10,
            output_tokens=5,
            latency_s=0.01,
            cost_usd=0.0,
        )
        return LLMResult(text=self.reply, usage=usage)


class FixedRetriever:
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, query, k=5):
        return self.passages[:k]


def test_rag_messages_include_ids_and_question():
    system, messages = rag_messages("कितना पैसा?", [make_passage("s0"), make_passage("s1")])
    assert system == SYSTEM_RAG
    content = messages[0]["content"]
    assert "[s0:benefits:0]" in content and "[s1:benefits:0]" in content
    assert content.endswith("Question: कितना पैसा?")


def test_parse_answer_valid_and_invalid():
    ok, err = parse_answer('{"text": "Rs 3000", "citations": ["a"], "refused": false}')
    assert err is None and ok.citations == ["a"]
    bad, err = parse_answer("Rs 3000 per year")
    assert err is not None and bad.text == "Rs 3000 per year" and not bad.refused


def test_closed_book_arm_drops_citations():
    reply = json.dumps({"text": "x", "citations": ["made-up:id:0"], "refused": False})
    model = EchoModel(reply)
    out = ClosedBookArm("arm1", model).answer("q?")
    assert out.answer.citations == [] and out.retrieved == []
    assert model.seen[0][0] == SYSTEM_CLOSED_BOOK


def test_rag_arm_records_retrieved_ids_and_usage():
    reply = json.dumps({"text": "x", "citations": ["s0:benefits:0"], "refused": False})
    retriever = FixedRetriever([make_passage(f"s{i}") for i in range(10)])
    out = RagArm("arm2", EchoModel(reply), retriever, k=3).answer("q?")
    assert out.retrieved == ["s0:benefits:0", "s1:benefits:0", "s2:benefits:0"]
    assert out.answer.citations == ["s0:benefits:0"]
    assert out.usage[0].input_tokens == 10 and out.latency_s >= 0


class ScriptedModel(EchoModel):
    """Replies with a search query to the rewrite prompt, and an answer otherwise."""

    def __init__(self, query_reply: str, answer_reply: str):
        super().__init__(answer_reply)
        self.query_reply = query_reply

    def chat(self, system, messages, schema=None, max_tokens=0):
        result = super().chat(system, messages, schema, max_tokens)
        if schema is not None and schema.__name__ == "SearchQuery":
            return LLMResult(text=self.query_reply, usage=result.usage)
        return result


class RecordingRetriever(FixedRetriever):
    def __init__(self, passages):
        super().__init__(passages)
        self.queries: list[str] = []

    def retrieve(self, query, k=5):
        self.queries.append(query)
        return super().retrieve(query, k)


ANSWER = json.dumps({"text": "x", "citations": [], "refused": False})


def test_query_translation_for_hindi_uses_english_query_but_original_question():
    from saathi.arms.prompts import SYSTEM_QUERY_REWRITE

    model = ScriptedModel(json.dumps({"query": "PMJJBY premium per year"}), ANSWER)
    retriever = RecordingRetriever([make_passage("s0")])
    q = "पीएमजेजेबीवाई का प्रीमियम कितना है?"
    out = RagArm("arm2b", model, retriever, translate_query=True).answer(q)
    assert retriever.queries == ["PMJJBY premium per year"]
    assert out.search_query == "PMJJBY premium per year"
    assert model.seen[0][0] == SYSTEM_QUERY_REWRITE
    assert q in model.seen[1][1][0]["content"]  # the answer prompt keeps the Hindi question
    assert len(out.usage) == 2  # rewrite call + answer call are both counted


def test_query_translation_skips_english_and_is_off_by_default():
    model = ScriptedModel(json.dumps({"query": "never used"}), ANSWER)
    retriever = RecordingRetriever([make_passage("s0")])
    out = RagArm("arm2b", model, retriever, translate_query=True).answer(
        "What is the premium for PMJJBY?"
    )
    assert retriever.queries == ["What is the premium for PMJJBY?"] and out.search_query is None
    assert len(model.seen) == 1
    retriever2 = RecordingRetriever([make_passage("s0")])
    RagArm("arm2", model, retriever2).answer("पीएमजेजेबीवाई का प्रीमियम कितना है?")
    assert retriever2.queries == ["पीएमजेजेबीवाई का प्रीमियम कितना है?"]


def test_query_translation_falls_back_to_question_on_bad_output():
    model = ScriptedModel("not json", ANSWER)
    retriever = RecordingRetriever([make_passage("s0")])
    q = "PMJJBY ka premium kitna hai?"
    out = RagArm("arm2b", model, retriever, translate_query=True).answer(q)
    assert retriever.queries == [q] and out.search_query is None


def test_old_predictions_without_search_query_still_load():
    from saathi.arms.arms import ArmOutput

    raw = {"answer": {"text": "x"}, "retrieved": [], "usage": [], "latency_s": 1.0}
    assert ArmOutput.model_validate(raw).search_query is None
