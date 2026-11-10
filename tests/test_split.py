import pytest

from saathi.data.split import (
    assign_by_scheme,
    check_group_split,
    check_scheme_split,
    split_by_group,
    split_schemes,
)
from saathi.schemas import Language
from tests.conftest import make_item

SCHEMES = [f"s{i}" for i in range(10)]


def test_scheme_split_is_disjoint_and_complete():
    train, test = split_schemes(SCHEMES, test_frac=0.3, seed=0)
    assert set(train) | set(test) == set(SCHEMES)
    assert not set(train) & set(test)
    assert len(test) == 3


def test_scheme_split_is_deterministic_and_order_independent():
    a = split_schemes(SCHEMES, 0.3, seed=1)
    b = split_schemes(list(reversed(SCHEMES)), 0.3, seed=1)
    assert a == b


def test_scheme_split_changes_with_seed():
    assert split_schemes(SCHEMES, 0.3, seed=0) != split_schemes(SCHEMES, 0.3, seed=1)


def test_scheme_split_keeps_both_sides_non_empty():
    train, test = split_schemes(["a", "b"], test_frac=0.9, seed=0)
    assert len(train) == 1 and len(test) == 1


@pytest.mark.parametrize("frac", [0, 1, -0.1])
def test_scheme_split_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        split_schemes(SCHEMES, frac, seed=0)


def test_assign_by_scheme_passes_leak_check(corpus):
    _, test_schemes = split_schemes(SCHEMES, 0.3, seed=0)
    train, test = assign_by_scheme(corpus, test_schemes)
    assert len(train) + len(test) == len(corpus)
    check_scheme_split(train, test)  # does not raise
    assert {it.scheme_id for it in test} == set(test_schemes)


def test_leak_check_catches_shared_scheme(corpus):
    train, test = assign_by_scheme(corpus, ["s0"])
    with pytest.raises(ValueError, match="scheme leak"):
        check_scheme_split(train + [test[0]], test)


def test_leak_check_catches_cited_passage_from_other_side(corpus):
    train, test = assign_by_scheme(corpus, ["s0"])
    bad = train[0].model_copy(update={"citations": ["s0:benefits:0"]})
    with pytest.raises(ValueError, match="cited passage leak"):
        check_scheme_split([bad] + train[1:], test)


def test_group_split_keeps_translations_together(corpus):
    train, test = split_by_group(corpus, test_frac=0.25, seed=0)
    check_group_split(train, test)  # does not raise
    for side in (train, test):
        by_group: dict[str, set[Language]] = {}
        for it in side:
            by_group.setdefault(it.group_id, set()).add(it.language)
        assert all(langs == set(Language) for langs in by_group.values())


def test_leak_check_catches_split_group():
    en = make_item("s0", 0, Language.EN)
    hi = make_item("s0", 0, Language.HI)
    with pytest.raises(ValueError, match="question group leak"):
        check_group_split([en], [hi])
