"""Combine generated Q&A and write the train/test splits.

Writes:
  data/qa/train.jsonl, data/qa/test.jsonl              primary split, by scheme
  data/qa/train_seen.jsonl, data/qa/test_seen.jsonl    secondary split, by question group
                                                       within the primary train schemes
  data/qa/split.json                                   scheme lists and seed

Usage:
    uv run python -m saathi.data.build_splits --test-frac 0.3 --seed 13
"""

import argparse
import json
from pathlib import Path

from saathi.data.split import (
    assign_by_scheme,
    check_group_split,
    check_scheme_split,
    split_by_group,
    split_schemes,
)
from saathi.io import read_jsonl, write_jsonl
from saathi.schemas import QAItem


def build(
    items: list[QAItem],
    test_frac: float,
    seed: int,
    out_dir: Path,
    force_test: list[str] | None = None,
) -> dict:
    """force_test: scheme IDs that must be in the test split (e.g. schemes with hand-written
    test questions), so nothing about them reaches the SFT data."""
    train_schemes, test_schemes = split_schemes([it.scheme_id for it in items], test_frac, seed)
    forced = sorted(set(force_test or []))
    if forced:
        test_schemes = sorted(set(test_schemes) | set(forced))
        train_schemes = sorted(set(train_schemes) - set(forced))
        if not train_schemes:
            raise ValueError("every scheme is forced into test; nothing left to train on")
    train, test = assign_by_scheme(items, test_schemes)
    check_scheme_split(train, test)
    train_seen, test_seen = split_by_group(train, test_frac=0.2, seed=seed)
    check_group_split(train_seen, test_seen)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "train_seen.jsonl", train_seen)
    write_jsonl(out_dir / "test_seen.jsonl", test_seen)
    summary = {
        "seed": seed,
        "test_frac": test_frac,
        "train_schemes": train_schemes,
        "test_schemes": test_schemes,
        "forced_test_schemes": forced,
        "counts": {
            "train": len(train),
            "test": len(test),
            "train_seen": len(train_seen),
            "test_seen": len(test_seen),
        },
    }
    (out_dir / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path, default=Path("data/qa/generated"))
    ap.add_argument("--out", type=Path, default=Path("data/qa"))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--extra-test",
        type=Path,
        nargs="*",
        default=[],
        help="QA files (e.g. data/qa/handwritten.jsonl) added to the pool; their schemes are "
        "forced into the test split",
    )
    args = ap.parse_args()
    items = [
        it
        for path in sorted(args.generated.glob("*.jsonl"))
        if path.name != "generation_log.jsonl"
        for it in read_jsonl(path, QAItem)
    ]
    extra = [it for path in args.extra_test for it in read_jsonl(path, QAItem)]
    summary = build(
        items + extra, args.test_frac, args.seed, args.out, sorted({it.scheme_id for it in extra})
    )
    print(json.dumps(summary["counts"]))


if __name__ == "__main__":
    main()
