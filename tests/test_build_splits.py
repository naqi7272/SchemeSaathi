import json

from saathi.data.build_splits import build
from saathi.io import read_jsonl
from saathi.schemas import QAItem


def test_build_writes_disjoint_splits(corpus, tmp_path):
    summary = build(corpus, test_frac=0.3, seed=0, out_dir=tmp_path)
    train = read_jsonl(tmp_path / "train.jsonl", QAItem)
    test = read_jsonl(tmp_path / "test.jsonl", QAItem)
    assert len(train) + len(test) == len(corpus)
    assert {it.scheme_id for it in test} == set(summary["test_schemes"])
    assert not {it.scheme_id for it in train} & {it.scheme_id for it in test}

    seen_train = read_jsonl(tmp_path / "train_seen.jsonl", QAItem)
    seen_test = read_jsonl(tmp_path / "test_seen.jsonl", QAItem)
    assert len(seen_train) + len(seen_test) == len(train)
    assert {it.scheme_id for it in seen_test} <= set(summary["train_schemes"])
    assert json.loads((tmp_path / "split.json").read_text())["seed"] == 0
