"""Aggregate scored (and judged) runs into tables, plots, and paired comparisons.

    uv run python -m saathi.eval.report --runs arm1-... arm2-... \
        --compare arm2-...:arm1-... arm4-...:arm2-...

Writes results/report.md, results/summary.json, results/plots/*.png.

Metric definitions are in docs/EVAL_METRICS.md (DRAFT until signed off). In short, per item:
  numbers_exact answerable, not refused, gold has numbers: EVERY gold number (amount, age,
                year) appears in the answer (code; the headline metric on the $0 path)
  refusal_ok    model refuses exactly when the question is unanswerable (code)
  ...           plus the other code metrics from score.py (citations, retrieval, script)
  accuracy      OPTIONAL judge: answerable items, judge says "correct" (refusal = wrong)
  faithful      OPTIONAL judge: every claim supported by the sources
The judge metrics are shown only if results/judge_validation.json shows Cohen's kappa
>= 0.6 overall and >= 0.4 in every language for that dimension; otherwise they are left out
of the report (with a note), even if judgments exist. No judge at all is fine.
CIs: 95% cluster bootstrap over question groups. Paired tests: cluster bootstrap on the
difference (primary) and exact McNemar (secondary); Holm correction across all tests.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from saathi.eval.judge import Judgment
from saathi.eval.run import RUNS_DIR, load_run_config
from saathi.eval.score import ItemScore
from saathi.eval.stats import CI, bootstrap_ci, holm, mcnemar_exact, paired_bootstrap
from saathi.io import read_jsonl
from saathi.schemas import Language, QAItem

LANGS = ["all", *[lang.value for lang in Language]]


def item_metrics(score: ItemScore, judgment: Judgment | None) -> dict[str, float | None]:
    """All per-item metric values; None = not applicable to this item."""
    judged_correct = None
    faithful = None
    if judgment is not None:
        judged_correct = float(judgment.verdict.correctness == "correct")
        faithful = float(judgment.verdict.faithful == "yes")

    def opt(x: bool | float | None) -> float | None:
        return None if x is None else float(x)

    accuracy = None
    if score.answerable:
        # refused answerable item -> wrong; otherwise needs a judgment
        accuracy = 0.0 if not score.refusal_correct else judged_correct
    numbers_exact = None
    if score.answerable and not score.refusal_correct:
        # refused an answerable item: wrong, if the gold answer has numbers to get right
        numbers_exact = 0.0 if score.has_gold_numbers else None
    elif score.number_recall is not None:
        numbers_exact = float(score.number_recall == 1.0)
    return {
        "numbers_exact": numbers_exact,
        "accuracy": accuracy,
        "faithful": faithful,
        "refusal_ok": float(score.refusal_correct),
        "refused_unanswerable": None if score.answerable else float(score.refusal_correct),
        "false_refusal": float(not score.refusal_correct) if score.answerable else None,
        "citation_precision": score.citation_precision,
        "citation_recall": score.citation_recall,
        "citations_grounded": opt(score.citations_grounded),
        "retrieval_recall": score.retrieval_recall,
        "number_recall": score.number_recall,
        "script_match": opt(score.script_match),
        "parse_error": float(score.parse_error),
        "latency_s": score.latency_s,
        "output_tokens": float(score.output_tokens),
        "cost_usd": score.cost_usd,
    }


METRICS = [
    "numbers_exact",
    "refused_unanswerable",
    "false_refusal",
    "citation_precision",
    "citation_recall",
    "citations_grounded",
    "retrieval_recall",
    "number_recall",
    "script_match",
    "parse_error",
    "accuracy",
    "faithful",
    "latency_s",
    "output_tokens",
    "cost_usd",
]
JUDGE_METRICS = {"accuracy": "kappa_correctness", "faithful": "kappa_faithful"}
PAIRED_METRICS = ["numbers_exact", "accuracy", "refusal_ok"]
BINARY_METRICS = {
    "numbers_exact",
    "accuracy",
    "faithful",
    "refusal_ok",
    "refused_unanswerable",
    "false_refusal",
    "citations_grounded",
    "script_match",
    "parse_error",
}


def judge_gate(path: Path, min_overall: float = 0.6, min_language: float = 0.4) -> dict[str, str]:
    """{judge metric: reason it is hidden}. Empty dict = every judge metric validated."""
    if not path.exists():
        return {m: f"no judge validation ({path} missing)" for m in JUDGE_METRICS}
    data = json.loads(path.read_text(encoding="utf-8"))
    hidden = {}
    for metric, key in JUDGE_METRICS.items():
        overall = (data.get("all") or {}).get(key)
        langs = {k: v.get(key) for k, v in data.items() if k != "all"}
        if overall is None or overall < min_overall:
            hidden[metric] = f"judge kappa {overall} < {min_overall} overall"
        elif any(v is None or v < min_language for v in langs.values()):
            low = {k: v for k, v in langs.items() if v is None or v < min_language}
            hidden[metric] = f"judge kappa below {min_language} for {low}"
    return hidden


def load_run(
    run_id: str, exclude_machine_made: bool = False
) -> tuple[dict[str, dict[str, float | None]], dict[str, QAItem]]:
    """{qa_id: metrics} for one run, and the run's items by qa_id."""
    cfg = load_run_config(run_id)
    items = {it.qa_id: it for it in read_jsonl(Path(cfg.test_file), QAItem)}
    if exclude_machine_made:
        items = {q: it for q, it in items.items() if not it.machine_made}
    run_dir = RUNS_DIR / run_id
    judgments = {}
    if (run_dir / "judgments.jsonl").exists():
        judgments = {j.qa_id: j for j in read_jsonl(run_dir / "judgments.jsonl", Judgment)}
    scores = [s for s in read_jsonl(run_dir / "scores.jsonl", ItemScore) if s.qa_id in items]
    unjudged = [
        s.qa_id for s in scores if s.answerable and s.refusal_correct and s.qa_id not in judgments
    ]
    if unjudged and judgments:
        print(
            f"WARNING {run_id}: {len(unjudged)} answers not judged; accuracy and "
            "faithfulness cover judged items only. Run `saathi-eval judge` first."
        )
    return {s.qa_id: item_metrics(s, judgments.get(s.qa_id)) for s in scores}, items


def summarize(
    per_item: dict[str, dict[str, float | None]], items: dict[str, QAItem]
) -> dict[str, dict[str, CI | None]]:
    """{language: {metric: CI}} with 'all' as a language. None when no applicable items."""
    out: dict[str, dict[str, CI | None]] = {}
    for lang in LANGS:
        out[lang] = {}
        ids = [q for q in per_item if lang == "all" or items[q].language == lang]
        for metric in METRICS:
            pairs = [
                (per_item[q][metric], items[q].group_id)
                for q in ids
                if per_item[q][metric] is not None
            ]
            out[lang][metric] = (
                bootstrap_ci([v for v, _ in pairs], [g for _, g in pairs]) if pairs else None
            )
    return out


def compare(
    a: dict[str, dict[str, float | None]],
    b: dict[str, dict[str, float | None]],
    items: dict[str, QAItem],
    metric: str,
) -> dict | None:
    """Paired comparison on items where both runs have a value for `metric`."""
    common = sorted(
        q for q in a if q in b and a[q][metric] is not None and b[q][metric] is not None
    )
    if not common:
        return None
    va = [a[q][metric] for q in common]
    vb = [b[q][metric] for q in common]
    res = paired_bootstrap(va, vb, [items[q].group_id for q in common])
    out = res.model_dump()
    if metric in BINARY_METRICS:
        n10, n01, p_mcnemar = mcnemar_exact([v == 1.0 for v in va], [v == 1.0 for v in vb])
        out |= {"a_only": n10, "b_only": n01, "p_mcnemar": p_mcnemar}
    return out


def fmt(ci: CI | None, pct: bool = True) -> str:
    if ci is None:
        return "n/a"
    if pct:
        return f"{100 * ci.mean:.1f} [{100 * ci.lo:.1f}, {100 * ci.hi:.1f}]"
    return f"{ci.mean:.2f} [{ci.lo:.2f}, {ci.hi:.2f}]"


PCT_METRICS = set(METRICS) - {"latency_s", "output_tokens", "cost_usd"}


def markdown(
    summaries: dict[str, dict], comparisons: list[dict], hidden: dict[str, str] | None = None
) -> str:
    hidden = hidden or {}
    lines = [
        "# Results",
        "",
        "Mean with 95% cluster-bootstrap CI. Percent unless noted. n/a = not applicable.",
        "Code-scored metrics come first; they need no judge.",
        "",
    ]
    for metric, reason in hidden.items():
        lines.append(f"- `{metric}` (LLM judge) not reported: {reason}.")
    if hidden:
        lines.append("")
    for metric in METRICS:
        if metric in hidden:
            continue
        unit = "%" if metric in PCT_METRICS else ("s" if metric == "latency_s" else "")
        lines += [
            f"## {metric}" + (f" ({unit})" if unit else ""),
            "",
            "| run | " + " | ".join(LANGS) + " |",
            "|---|" + "---|" * len(LANGS),
        ]
        for run_id, summary in summaries.items():
            cells = [fmt(summary[lang][metric], metric in PCT_METRICS) for lang in LANGS]
            lines.append(f"| {run_id} | " + " | ".join(cells) + " |")
        lines.append("")
    if comparisons:
        lines += [
            "## Paired comparisons",
            "",
            "diff = A - B (percentage points). p_holm: cluster-bootstrap p, Holm-adjusted "
            "across all rows. McNemar p is unadjusted, for reference.",
            "",
            "| A | B | metric | language | n | diff [95% CI] | p_holm | McNemar p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in comparisons:
            mc = f"{c['p_mcnemar']:.3g}" if "p_mcnemar" in c else "-"
            lines.append(
                f"| {c['a']} | {c['b']} | {c['metric']} | {c['language']} | {c['n']} | "
                f"{100 * c['diff']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] | "
                f"{c['p_holm']:.3g} | {mc} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- plots
SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def plot_metric(summaries: dict[str, dict], metric: str, path: Path) -> None:
    """Small multiples, one panel per language: each run is a dot with its 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(summaries)
    langs = [lang for lang in LANGS if lang != "all"]
    fig, axes = plt.subplots(
        1,
        len(langs),
        figsize=(4 * len(langs), 0.5 * len(runs) + 1.4),
        sharey=True,
        facecolor=SURFACE,
    )
    scale = 100 if metric in PCT_METRICS else 1
    for ax, lang in zip(np.atleast_1d(axes), langs, strict=True):
        ax.set_facecolor(SURFACE)
        for y, run in enumerate(runs):
            ci = summaries[run][lang][metric]
            if ci is None:
                continue
            ax.plot(
                [ci.lo * scale, ci.hi * scale], [y, y], color=SERIES, lw=2, solid_capstyle="round"
            )
            ax.plot(ci.mean * scale, y, "o", color=SERIES, ms=8, mec=SURFACE, mew=2)
            ax.annotate(
                f"{ci.mean * scale:.0f}",
                (ci.mean * scale, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_2,
            )
        ax.set_title(lang, color=INK, fontsize=11, loc="left")
        ax.set_yticks(range(len(runs)), runs, color=INK, fontsize=9)
        ax.invert_yaxis()
        if metric in PCT_METRICS:
            ax.set_xlim(-3, 103)  # margin so whiskers and dots at 0 or 100 are not clipped
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    unit = " (%)" if metric in PCT_METRICS else ""
    fig.suptitle(f"{metric}{unit}, mean and 95% CI", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(w_pad=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_report(
    run_ids: list[str],
    pairs: list[tuple[str, str]],
    out_dir: Path,
    loader: Callable = load_run,
    hidden: dict[str, str] | None = None,
) -> dict:
    """hidden: judge metrics to leave out ({metric: reason}); main() fills it from
    judge_gate(). None/{} shows everything (used by tests)."""
    hidden = hidden or {}
    per_run, items = {}, {}
    for run_id in run_ids:
        per_run[run_id], run_items = loader(run_id)
        items.update(run_items)
    summaries = {r: summarize(per_run[r], items) for r in run_ids}

    comparisons = []
    for a, b in pairs:
        for metric in [m for m in PAIRED_METRICS if m not in hidden]:
            for lang in LANGS:
                ids = {q for q in per_run[a] if lang == "all" or items[q].language == lang}
                res = compare(
                    {q: per_run[a][q] for q in ids},
                    {q: v for q, v in per_run[b].items() if q in ids},
                    items,
                    metric,
                )
                if res:
                    comparisons.append({"a": a, "b": b, "metric": metric, "language": lang} | res)
    for c, p in zip(comparisons, holm([c["p"] for c in comparisons]), strict=True):
        c["p_holm"] = p

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown(summaries, comparisons, hidden), encoding="utf-8")
    summary_json = {
        r: {
            lang: {m: (ci.model_dump() if ci else None) for m, ci in s.items() if m not in hidden}
            for lang, s in summ.items()
        }
        for r, summ in summaries.items()
    }
    result = {"summaries": summary_json, "comparisons": comparisons, "hidden": hidden}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for metric in (
        "numbers_exact",
        "accuracy",
        "faithful",
        "refused_unanswerable",
        "citation_precision",
        "latency_s",
    ):
        if metric in hidden:
            continue
        plot_metric(summaries, metric, out_dir / "plots" / f"{metric}.png")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="A:B pairs of run ids")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--judge-validation", type=Path, default=Path("results/judge_validation.json"))
    ap.add_argument(
        "--exclude-machine-made",
        action="store_true",
        help="leave out machine-made items (e.g. romanized variants); report them separately",
    )
    args = ap.parse_args()
    pairs = [tuple(p.split(":", 1)) for p in args.compare]
    hidden = judge_gate(args.judge_validation)
    for metric, reason in hidden.items():
        print(f"{metric}: not reported ({reason})")

    def loader(run_id: str):
        return load_run(run_id, args.exclude_machine_made)

    build_report(args.runs, pairs, args.out, loader=loader, hidden=hidden)
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
