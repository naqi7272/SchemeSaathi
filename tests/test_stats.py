import pytest

from saathi.eval.stats import bootstrap_ci, cohen_kappa, holm, mcnemar_exact, paired_bootstrap


def test_bootstrap_ci_constant_values_has_zero_width():
    ci = bootstrap_ci([1.0] * 20)
    assert (ci.mean, ci.lo, ci.hi, ci.n) == (1.0, 1.0, 1.0, 20)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 5
    a, b = bootstrap_ci(values, seed=3), bootstrap_ci(values, seed=3)
    assert a == b
    assert a.lo < a.mean == 0.7 < a.hi


def test_bootstrap_ci_roughly_matches_normal_approximation():
    # p=0.5, n=400 -> standard error 0.025 -> 95% CI about +-0.049
    ci = bootstrap_ci([1, 0] * 200, n_boot=20_000)
    assert ci.hi - ci.lo == pytest.approx(0.098, abs=0.01)


def test_cluster_bootstrap_is_wider_for_correlated_items():
    # 40 groups, all three items in a group share one outcome
    values = [g % 2 for g in range(40) for _ in range(3)]
    clusters = [str(g) for g in range(40) for _ in range(3)]
    naive = bootstrap_ci(values)
    clustered = bootstrap_ci(values, clusters)
    assert clustered.hi - clustered.lo > 1.5 * (naive.hi - naive.lo)


def test_paired_bootstrap_detects_clear_difference():
    a = [1.0] * 50
    b = [0.0] * 10 + [1.0] * 40
    res = paired_bootstrap(a, b, [str(i) for i in range(50)])
    assert res.diff == pytest.approx(0.2)
    assert 0 < res.lo <= res.diff <= res.hi
    assert res.p < 0.01


def test_paired_bootstrap_no_difference():
    a = [1, 0, 1, 0] * 10
    res = paired_bootstrap(a, a, [str(i) for i in range(40)])
    assert res.diff == 0 and res.p == 1.0


def test_paired_bootstrap_rejects_misaligned_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1], ["a", "b"])


def test_mcnemar_exact_textbook_value():
    # 1 pair where only a is right, 6 where only b is right:
    # p = 2 * (C(7,0) + C(7,1)) / 2^7 = 16/128
    a = [True] + [False] * 6 + [True] * 5
    b = [False] + [True] * 6 + [True] * 5
    assert mcnemar_exact(a, b) == (1, 6, 0.125)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_mcnemar_p_capped_at_one():
    assert mcnemar_exact([True, False], [False, True])[2] == 1.0


def test_holm_known_values():
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []


def test_cohen_kappa_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po=0.7, pe=0.5, kappa=0.4
    x = ["yes"] * 25 + ["no"] * 25
    y = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(x, y) == pytest.approx(0.4)


def test_cohen_kappa_edge_cases():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])
