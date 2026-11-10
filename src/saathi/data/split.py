"""Train/test splits.

Primary split: by scheme. No scheme, and therefore no passage, appears on both sides.
Secondary split: within the training schemes, by question group. Used only to ask whether
fine-tuning injects knowledge about schemes the model has seen. Translations of one question
share a group_id and always land on the same side.
"""

import random

from saathi.schemas import QAItem


def split_schemes(
    scheme_ids: list[str], test_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Return (train_scheme_ids, test_scheme_ids), both sorted."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    ids = sorted(set(scheme_ids))
    if len(ids) < 2:
        raise ValueError("need at least two schemes to split")
    random.Random(seed).shuffle(ids)
    n_test = min(len(ids) - 1, max(1, round(len(ids) * test_frac)))
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def assign_by_scheme(
    items: list[QAItem], test_schemes: list[str]
) -> tuple[list[QAItem], list[QAItem]]:
    test_set = set(test_schemes)
    train = [it for it in items if it.scheme_id not in test_set]
    test = [it for it in items if it.scheme_id in test_set]
    return train, test


def split_by_group(
    items: list[QAItem], test_frac: float, seed: int
) -> tuple[list[QAItem], list[QAItem]]:
    """Secondary split: hold out whole question groups."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    groups = sorted({it.group_id for it in items})
    random.Random(seed).shuffle(groups)
    n_test = round(len(groups) * test_frac)
    test_groups = set(groups[:n_test])
    train = [it for it in items if it.group_id not in test_groups]
    test = [it for it in items if it.group_id in test_groups]
    return train, test


def check_scheme_split(train: list[QAItem], test: list[QAItem]) -> None:
    """Raise if any scheme, cited passage, or question group appears on both sides."""
    _check_disjoint("scheme", {it.scheme_id for it in train}, {it.scheme_id for it in test})
    check_group_split(train, test)
    train_cites = {c for it in train for c in it.citations}
    test_cites = {c for it in test for c in it.citations}
    _check_disjoint("cited passage", train_cites, test_cites)


def check_group_split(train: list[QAItem], test: list[QAItem]) -> None:
    """Raise if any question group (a question and its translations) appears on both sides."""
    _check_disjoint("question group", {it.group_id for it in train}, {it.group_id for it in test})


def _check_disjoint(what: str, a: set[str], b: set[str]) -> None:
    overlap = a & b
    if overlap:
        raise ValueError(f"{what} leak across split: {sorted(overlap)[:5]}")
