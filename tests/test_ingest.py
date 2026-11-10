import httpx
import pytest
import yaml

from saathi.data.ingest import (
    SourceDoc,
    SourceEntry,
    blocked_reason,
    fetch_all,
    html_sections,
    load_sources,
    markdown_sections,
    process,
    slug,
)
from saathi.io import read_jsonl
from saathi.schemas import ManifestEntry, Passage, Scheme

PAGE = """<html><head><script>var x=1;</script></head><body>
<nav>Home | About</nav>
<h1>Toy Scheme</h1><p>An example scheme.</p>
<h2>Eligibility</h2><ul><li>Farmers with up to <b>2 ha</b> land</li><li>Age 18-60</li></ul>
<h2>Benefits</h2><p>Rs 3,000 per year.</p>
<footer>Copyright</footer></body></html>"""


def fake_get(pages: dict[str, str], calls: list[str]):
    def get(url):
        calls.append(url)
        if url in pages:
            return httpx.Response(
                200,
                text=pages[url],
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(404, request=httpx.Request("GET", url))

    return get


def entry(url, **doc):
    return SourceEntry(
        scheme_id="toy", name="Toy", terms_note="test terms", docs=[SourceDoc(url=url, **doc)]
    )


def test_blocked_reason_covers_subdomains():
    assert blocked_reason("https://www.myscheme.gov.in/schemes/x")
    assert blocked_reason("https://api.myscheme.gov.in/x")
    assert blocked_reason("https://pmkisan.gov.in/") is None
    assert blocked_reason("https://notmyscheme.gov.in/") is None


def test_fetch_refuses_blocked_domain_without_permission(tmp_path):
    calls: list[str] = []
    with pytest.raises(PermissionError, match="Terms of Use"):
        fetch_all(
            [entry("https://www.myscheme.gov.in/schemes/toy")],
            tmp_path,
            get=fake_get({}, calls),
            sleep=lambda s: None,
        )
    assert calls == []  # nothing was requested


def test_fetch_accepts_hand_saved_copy_of_blocked_page(tmp_path):
    saved = tmp_path / "saved.html"
    saved.write_text(PAGE, encoding="utf-8")
    calls: list[str] = []
    new = fetch_all(
        [entry("https://www.myscheme.gov.in/schemes/toy", file=str(saved))],
        tmp_path / "raw",
        get=fake_get({}, calls),
        sleep=lambda s: None,
    )
    assert calls == [] and len(new) == 1
    assert new[0].raw_path.startswith("toy/") and new[0].raw_path.endswith(".html")


def test_fetch_respects_robots_txt_and_continues(tmp_path, capsys):
    pages = {"https://x.gov.in/robots.txt": "User-agent: *\nDisallow: /private/"}
    calls: list[str] = []
    new = fetch_all(
        [entry("https://x.gov.in/private/toy")],
        tmp_path,
        get=fake_get(pages, calls),
        sleep=lambda s: None,
    )
    assert new == [] and calls == ["https://x.gov.in/robots.txt"]  # page never requested
    out = capsys.readouterr().out
    assert "disallowed by robots.txt" in out and "save it as" in out


def test_fetch_records_manifest_and_skips_done(tmp_path):
    pages = {"https://x.gov.in/toy": PAGE}
    calls: list[str] = []
    sleeps: list[float] = []
    fetch_all(
        [entry("https://x.gov.in/toy")], tmp_path, get=fake_get(pages, calls), sleep=sleeps.append
    )
    manifest = read_jsonl(tmp_path / "manifest.jsonl", ManifestEntry)
    assert len(manifest) == 1 and manifest[0].terms_note == "test terms"
    assert (tmp_path / manifest[0].raw_path).read_text(encoding="utf-8") == PAGE
    assert sleeps and all(s > 0 for s in sleeps)  # polite delay
    calls.clear()
    assert (
        fetch_all(
            [entry("https://x.gov.in/toy")],
            tmp_path,
            get=fake_get(pages, calls),
            sleep=lambda s: None,
        )
        == []
    )
    assert calls == []


def test_html_sections_drops_boilerplate_and_splits_on_headings():
    sections = dict(html_sections(PAGE))
    assert list(sections) == ["Toy Scheme", "Eligibility", "Benefits"]
    assert sections["Eligibility"] == "Farmers with up to 2 ha land\n\nAge 18-60"
    assert "Home" not in " ".join(sections.values())
    assert "var x" not in " ".join(sections.values())


def test_markdown_sections_keep_hindi_text():
    text = "# पात्रता\nकिसान परिवार\n\n## लाभ\n₹6000\n"
    assert markdown_sections(text) == [("पात्रता", "किसान परिवार"), ("लाभ", "₹6000")]


def test_slug_falls_back_for_non_latin_headings():
    assert slug("How to Apply?") == "how-to-apply"
    assert slug("पात्रता") == "section"


def test_process_builds_passages_with_source_urls(tmp_path):
    md = tmp_path / "toy.md"
    md.write_text("# Eligibility\nSmall farmers.\n# Benefits\nRs 3,000.\n", encoding="utf-8")
    html = tmp_path / "toy.html"
    html.write_text(PAGE, encoding="utf-8")
    e = SourceEntry(
        scheme_id="toy",
        name="Toy",
        terms_note="t",
        docs=[
            SourceDoc(url="https://a.gov.in/guidelines", file=str(md)),
            SourceDoc(url="https://b.gov.in/page", file=str(html)),
        ],
    )
    raw, out = tmp_path / "raw", tmp_path / "out"
    fetch_all([e], raw, get=fake_get({}, []), sleep=lambda s: None)
    process([e], raw, out)
    scheme = read_jsonl(out / "schemes.jsonl", Scheme)[0]
    assert list(scheme.sections)[:2] == ["d0-eligibility", "d0-benefits"]
    assert "d1-eligibility" in scheme.sections
    passages = read_jsonl(out / "passages.jsonl", Passage)
    by_id = {p.passage_id: p for p in passages}
    assert by_id["toy:d0-eligibility:0"].source_url == "https://a.gov.in/guidelines"
    assert by_id["toy:d1-benefits:0"].source_url == "https://b.gov.in/page"


def test_load_sources_rejects_duplicates_and_bad_ids(tmp_path):
    path = tmp_path / "s.yaml"
    item = {"scheme_id": "toy", "name": "T", "terms_note": "t", "docs": [{"url": "u"}]}
    path.write_text(yaml.safe_dump([item, item]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_sources(path)
    path.write_text(yaml.safe_dump([item | {"scheme_id": "Bad ID"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="slug"):
        load_sources(path)


def test_fetch_skips_network_errors_and_continues(tmp_path):
    calls: list[str] = []

    def get(url):
        calls.append(url)
        if "broken" in url:
            raise httpx.ConnectError("certificate verify failed")
        if url.endswith("robots.txt"):
            return httpx.Response(404, request=httpx.Request("GET", url))
        return httpx.Response(200, text=PAGE, request=httpx.Request("GET", url))

    entries = [
        SourceEntry(
            scheme_id="bad",
            name="Bad",
            terms_note="t",
            docs=[SourceDoc(url="https://broken.gov.in/x")],
        ),
        SourceEntry(
            scheme_id="good",
            name="Good",
            terms_note="t",
            docs=[SourceDoc(url="https://fine.gov.in/x")],
        ),
    ]
    new = fetch_all(entries, tmp_path, get=get, sleep=lambda s: None)
    assert [m.scheme_id for m in new] == ["good"]
    # the failed document is not in the manifest, so a rerun retries it
    assert [m.scheme_id for m in read_jsonl(tmp_path / "manifest.jsonl", ManifestEntry)] == ["good"]


DIV_PAGE = """<html><body><div class="hero"><span>PM Toy Yojana</span></div>
<div class="content"><h2>Eligibility</h2>
<div>All <span>landholding</span> farmer families.</div>
<div>Age 18 or above.<br>Must have Aadhaar.</div>
<h2>Benefits</h2><div><div>Rs 6,000 per year in <b>three</b> instalments.</div></div></div>
</body></html>"""


def test_html_sections_reads_div_and_span_text():
    sections = dict(html_sections(DIV_PAGE))
    assert list(sections) == ["intro", "Eligibility", "Benefits"]
    assert sections["intro"] == "PM Toy Yojana"
    assert sections["Eligibility"].split("\n\n") == [
        "All landholding farmer families.",
        "Age 18 or above.",
        "Must have Aadhaar.",
    ]
    assert sections["Benefits"] == "Rs 6,000 per year in three instalments."


def test_empty_saved_file_is_rejected_not_recorded(tmp_path):
    empty = tmp_path / "empty.html"
    empty.write_text("\n", encoding="utf-8")
    new = fetch_all(
        [entry("https://x.gov.in/toy", file=str(empty))],
        tmp_path / "raw",
        get=fake_get({}, []),
        sleep=lambda s: None,
    )
    assert new == [] and not (tmp_path / "raw" / "manifest.jsonl").exists()


def test_changed_saved_file_is_copied_again_and_latest_wins(tmp_path):
    saved = tmp_path / "page.html"
    raw, out = tmp_path / "raw", tmp_path / "out"
    e = entry("https://x.gov.in/toy", file=str(saved))
    saved.write_text(PAGE, encoding="utf-8")
    assert len(fetch_all([e], raw, get=fake_get({}, []), sleep=lambda s: None)) == 1
    assert fetch_all([e], raw, get=fake_get({}, []), sleep=lambda s: None) == []  # unchanged
    saved.write_text(DIV_PAGE, encoding="utf-8")
    assert len(fetch_all([e], raw, get=fake_get({}, []), sleep=lambda s: None)) == 1
    process([e], raw, out)
    scheme = read_jsonl(out / "schemes.jsonl", Scheme)[0]
    assert "d0-benefits" in scheme.sections and "d1-benefits" not in scheme.sections
    assert "Rs 6,000" in scheme.sections["d0-benefits"]


def test_process_warns_when_document_has_no_text(tmp_path, capsys):
    blank = tmp_path / "blank.html"
    blank.write_text(
        "<html><head><script>" + "x" * 100 + "</script></head><body></body></html>",
        encoding="utf-8",
    )
    e = entry("https://x.gov.in/toy", file=str(blank))
    fetch_all([e], tmp_path / "raw", get=fake_get({}, []), sleep=lambda s: None)
    process([e], tmp_path / "raw", tmp_path / "out")
    assert "WARNING toy: no text" in capsys.readouterr().out


def test_failed_download_suggests_name_and_manual_copy_is_used(tmp_path, capsys):
    def get(url):
        raise httpx.ConnectError("root certificate not trusted")

    manual = tmp_path / "manual"
    e = SourceEntry(
        scheme_id="toy",
        name="Toy",
        terms_note="t",
        docs=[
            SourceDoc(url="https://x.gov.in/a.html"),
            SourceDoc(url="https://x.gov.in/Guide.PDF"),
        ],
    )
    raw = tmp_path / "raw"
    assert fetch_all([e], raw, get=get, sleep=lambda s: None, manual_dir=manual) == []
    out = capsys.readouterr().out
    assert f"{manual.as_posix()}/toy-0.html" in out and f"{manual.as_posix()}/toy-1.pdf" in out
    manual.mkdir()
    (manual / "toy-0.html").write_text(PAGE, encoding="utf-8")
    new = fetch_all([e], raw, get=get, sleep=lambda s: None, manual_dir=manual)
    assert [(m.url, m.raw_path.endswith(".html")) for m in new] == [
        ("https://x.gov.in/a.html", True)
    ]


def test_shipped_sources_yaml_is_valid():
    from pathlib import Path

    entries = load_sources(Path("data/sources.yaml"))
    assert len(entries) >= 25
    assert all(e.terms_note and e.docs for e in entries)
    for e in entries:  # myScheme pages must be hand-saved, never auto-fetched
        assert all(d.file or "myscheme.gov.in" not in d.url for d in e.docs)


def test_stale_and_missing_manifest_records_are_handled(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "out"
    pages = {"https://x.gov.in/old": PAGE, "https://x.gov.in/new": DIV_PAGE}
    old = entry("https://x.gov.in/old")
    fetch_all([old], raw, get=fake_get(pages, []), sleep=lambda s: None)
    # sources.yaml now lists a different URL, and the old raw file was deleted by hand
    for m in read_jsonl(raw / "manifest.jsonl", ManifestEntry):
        (raw / m.raw_path).unlink()
    new = entry("https://x.gov.in/new")
    fetch_all([new], raw, get=fake_get(pages, []), sleep=lambda s: None)
    process([new], raw, out)  # must not crash on the stale record
    scheme = read_jsonl(out / "schemes.jsonl", Scheme)[0]
    assert "d0-benefits" in scheme.sections and scheme.source_url == "https://x.gov.in/new"


def test_missing_raw_file_is_fetched_again_and_skips_are_printed(tmp_path, capsys):
    raw = tmp_path / "raw"
    pages = {"https://x.gov.in/toy": PAGE}
    e = entry("https://x.gov.in/toy")
    fetch_all([e], raw, get=fake_get(pages, []), sleep=lambda s: None)
    fetch_all([e], raw, get=fake_get(pages, []), sleep=lambda s: None)
    assert "have toy https://x.gov.in/toy" in capsys.readouterr().out
    for m in read_jsonl(raw / "manifest.jsonl", ManifestEntry):
        (raw / m.raw_path).unlink()
    assert len(fetch_all([e], raw, get=fake_get(pages, []), sleep=lambda s: None)) == 1


def test_fetch_rejects_pdf_url_that_returns_a_web_page(tmp_path, capsys):
    pages = {"https://x.gov.in/guide.pdf": PAGE}  # served as HTML: an error page
    assert fetch_all([entry("https://x.gov.in/guide.pdf")], tmp_path, get=fake_get(pages, []),
                     sleep=lambda s: None, manual_dir=tmp_path / "m") == []  # fmt: skip
    assert "did not send a PDF" in capsys.readouterr().out


def test_fetch_rejects_bad_manual_pdf(tmp_path, capsys):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "toy-0.pdf").write_bytes(b"<!DOCTYPE html>" + b"x" * 500)
    e = entry("https://x.gov.in/guide.pdf")
    assert fetch_all([e], tmp_path / "raw", get=fake_get({}, []), sleep=lambda s: None,
                     manual_dir=manual) == []  # fmt: skip
    assert "BAD toy" in capsys.readouterr().out


def test_process_skips_broken_file_and_keeps_going(tmp_path, capsys):
    raw = tmp_path / "raw"
    (raw / "toy").mkdir(parents=True)
    (raw / "toy" / "broken.pdf").write_bytes(b"%PDF-1.4\n" + b"\x00" * 200)  # cut off
    (raw / "toy" / "page.pdf").write_text(PAGE, encoding="utf-8")  # HTML with a .pdf name
    good = tmp_path / "ok.md"
    good.write_text("# Benefits\nRs 3,000 per year.\n", encoding="utf-8")
    e = SourceEntry(
        scheme_id="toy",
        name="Toy",
        terms_note="t",
        docs=[SourceDoc(url="https://x.gov.in/a.pdf"), SourceDoc(url="https://x.gov.in/b.pdf")],
    )
    from datetime import UTC, datetime

    from saathi.io import write_jsonl

    write_jsonl(raw / "manifest.jsonl", [
        ManifestEntry(scheme_id="toy", url=u, fetched_at=datetime.now(UTC), sha256="x",
                      raw_path=f"toy/{f}", terms_note="t")
        for u, f in [("https://x.gov.in/a.pdf", "broken.pdf"),
                     ("https://x.gov.in/b.pdf", "page.pdf")]
    ])  # fmt: skip
    process([e], raw, tmp_path / "out")
    out = capsys.readouterr().out
    assert "BROKEN toy: toy/broken.pdf" in out and "1 schemes" in out
    scheme = read_jsonl(tmp_path / "out" / "schemes.jsonl", Scheme)[0]
    assert any("3,000" in t for t in scheme.sections.values())  # the HTML one was read
