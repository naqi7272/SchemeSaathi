"""Collect scheme documents from a hand-curated source list and turn them into passages.

Sources come only from data/sources.yaml, so every document is one a human chose and
whose terms were checked. Each entry is either a URL to fetch or a file saved by hand.

Fetching rules:
- Domains in BLOCKED_DOMAINS are never fetched automatically. myScheme's Terms of Use
  prohibit bots and scrapers without written permission; an entry for a blocked domain
  must set `written_permission` (a reference to the permission) or use a saved `file`.
- robots.txt is respected, requests are spaced by DELAY_S, and the user agent names
  the project.
- Every fetched or copied file is stored under data/raw/<scheme_id>/ and recorded in
  data/raw/manifest.jsonl with URL, fetch time, sha256, and the terms note.

Usage:
    uv run python -m saathi.data.ingest fetch    # download URLs listed in sources.yaml
    uv run python -m saathi.data.ingest process  # raw files -> schemes.jsonl, passages.jsonl
"""

import argparse
import hashlib
import re
import ssl
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import truststore
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, model_validator

from saathi.data.chunk import chunk_scheme
from saathi.io import append_jsonl, read_jsonl, write_jsonl
from saathi.schemas import ManifestEntry, Scheme

USER_AGENT = "SchemeSaathi/0.1 (research; github.com/naqi7272/SchemeSaathi)"
DELAY_S = 3.0
MIN_BYTES = 20  # anything smaller is an empty or failed save
BLOCKED_DOMAINS = {
    "myscheme.gov.in": "Terms of Use prohibit bots/scrapers without written permission",
}


class SourceDoc(BaseModel):
    url: str  # canonical URL, cited in answers even when the file was saved by hand
    file: str | None = None  # path to a hand-saved copy (html, pdf, txt, md)
    written_permission: str | None = None


class SourceEntry(BaseModel):
    scheme_id: str
    name: str
    terms_note: str  # which terms of use apply, checked by a human
    docs: list[SourceDoc]

    @model_validator(mode="after")
    def _check_id(self) -> "SourceEntry":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.scheme_id):
            raise ValueError(f"scheme_id must be lowercase slug, got {self.scheme_id!r}")
        return self


def load_sources(path: Path) -> list[SourceEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    entries = [SourceEntry.model_validate(e) for e in data]
    ids = [e.scheme_id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scheme_id in sources")
    return entries


def blocked_reason(url: str) -> str | None:
    host = urlparse(url).hostname or ""
    for domain, reason in BLOCKED_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return reason
    return None


def robots_allows(url: str, get: Callable[[str], httpx.Response]) -> bool:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = get(robots_url)
    if response.status_code >= 400:
        return True  # no robots.txt
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def _suffix(url: str, content_type: str) -> str:
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return ".pdf"
    return ".html"


def fetch_all(
    entries: list[SourceEntry],
    raw_dir: Path,
    get: Callable[[str], httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    manual_dir: Path = Path("data/manual"),
) -> list[ManifestEntry]:
    """Fetch or copy every doc not already in the manifest. Returns new manifest entries.

    A document is read from a hand-saved file when `file:` is set, or when a file named
    data/manual/<scheme_id>-<doc number>.<pdf|html|htm|txt|md> exists (doc number counts
    from 0 in sources.yaml). So when a download fails, save it in the browser under the
    printed name and rerun; no need to edit sources.yaml."""
    manifest_path = raw_dir / "manifest.jsonl"
    # latest record per (scheme, url) whose raw file still exists; anything else is redone,
    # and a hand-saved file is copied again when its content changes
    done = {}
    if manifest_path.exists():
        for m in read_jsonl(manifest_path, ManifestEntry):
            if (raw_dir / m.raw_path).exists():
                done[(m.scheme_id, m.url)] = m.sha256
    if get is None:
        # Verify certificates with the operating system's trust store (Windows/macOS/Linux)
        # rather than Python's bundled list: some government sites send incomplete
        # certificate chains that the OS can complete. Verification is never turned off.
        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30, verify=verify
        )
        get = client.get
    new = []
    failed = []
    for entry in entries:
        for i, doc in enumerate(entry.docs):
            key = (entry.scheme_id, doc.url)
            file = doc.file or _manual_copy(manual_dir, entry.scheme_id, i)
            if file:
                content = Path(file).read_bytes()
                suffix = Path(file).suffix or ".txt"
                if done.get(key) == hashlib.sha256(content).hexdigest():
                    print(f"have {entry.scheme_id} {doc.url} (from {Path(file).as_posix()})")
                    continue
                if suffix == ".pdf" and not looks_like_pdf(content):
                    print(
                        f"BAD {entry.scheme_id} {file}: not a real PDF (a web page or an "
                        "incomplete download). Download it again from the browser."
                    )
                    failed.append(doc.url)
                    continue
                if len(content.strip()) < MIN_BYTES:
                    print(
                        f"EMPTY {entry.scheme_id} {file}: only {len(content)} bytes. "
                        "Save the page again from the browser (Ctrl+S) after it has loaded."
                    )
                    failed.append(doc.url)
                    continue
            elif key in done:
                print(f"have {entry.scheme_id} {doc.url}")
                continue
            else:
                reason = blocked_reason(doc.url)
                if reason and not doc.written_permission:
                    raise PermissionError(
                        f"{doc.url}: {reason}. Save the page by hand "
                        "(file:) or record written_permission."
                    )
                ext = ".pdf" if doc.url.lower().endswith(".pdf") else ".html"
                hint = (manual_dir / f"{entry.scheme_id}-{i}{ext}").as_posix()
                try:
                    allowed = robots_allows(doc.url, get)
                except httpx.HTTPError as err:
                    print(f"FAILED {entry.scheme_id} {doc.url}: {err}\n       -> save it as {hint}")
                    failed.append(doc.url)
                    continue
                if not allowed:  # the bot stays out; a person may still save the page
                    print(
                        f"SKIPPED {entry.scheme_id} {doc.url}: disallowed by robots.txt\n"
                        f"       -> open it in your browser and save it as {hint}"
                    )
                    failed.append(doc.url)
                    continue
                try:
                    sleep(DELAY_S)
                    response = get(doc.url)
                    response.raise_for_status()
                except httpx.HTTPError as err:  # network/TLS/HTTP error: report, try the rest
                    print(
                        f"FAILED {entry.scheme_id} {doc.url}: {err}\n"
                        f"       -> open it in your browser and save it as {hint}"
                    )
                    failed.append(doc.url)
                    continue
                content = response.content
                suffix = _suffix(doc.url, response.headers.get("content-type", ""))
                sleep(DELAY_S)
                if suffix == ".pdf" and not looks_like_pdf(content):
                    print(
                        f"FAILED {entry.scheme_id} {doc.url}: the server did not send a PDF "
                        f"(probably an error page)\n       -> open it in your browser and save "
                        f"it as {hint}"
                    )
                    failed.append(doc.url)
                    continue
                if len(content.strip()) < MIN_BYTES:
                    print(f"EMPTY {entry.scheme_id} {doc.url}: only {len(content)} bytes")
                    failed.append(doc.url)
                    continue
            sha = hashlib.sha256(content).hexdigest()
            out = raw_dir / entry.scheme_id / f"{sha[:16]}{suffix}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            record = ManifestEntry(
                scheme_id=entry.scheme_id,
                url=doc.url,
                fetched_at=datetime.now(UTC),
                sha256=sha,
                raw_path=out.relative_to(raw_dir).as_posix(),
                terms_note=entry.terms_note
                + (f" | permission: {doc.written_permission}" if doc.written_permission else ""),
            )
            append_jsonl(manifest_path, record)
            new.append(record)
            print(f"ok {entry.scheme_id} {doc.url}")
    if failed:
        print(f"{len(failed)} document(s) failed. Save each under the name shown, then rerun.")
    return new


MANUAL_SUFFIXES = (".pdf", ".html", ".htm", ".txt", ".md")


def _manual_copy(manual_dir: Path, scheme_id: str, index: int) -> Path | None:
    for suffix in MANUAL_SUFFIXES:
        path = manual_dir / f"{scheme_id}-{index}{suffix}"
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------- text extraction


def slug(text: str, max_len: int = 40) -> str:
    """ASCII slug for section names; non-Latin headings fall back to 'section'."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "section"


HEADINGS = ["h1", "h2", "h3", "h4"]
BLOCKS = [
    "p",
    "div",
    "li",
    "td",
    "th",
    "tr",
    "section",
    "article",
    "ul",
    "ol",
    "table",
    "dd",
    "dt",
    "blockquote",
    "pre",
    "main",
]
HEADING_MARK = "\x00HEADING\x00"


def html_sections(html: str) -> list[tuple[str, str]]:
    """Split all visible page text into (heading, text) sections at h1-h4 headings.

    Text in any element counts (many sites use <div>/<span>, not <p>). Block elements end
    a line; inline elements (<b>, <a>, <span>) stay inside their sentence."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        tag.decompose()
    body = soup.body or soup
    for br in body.find_all("br"):
        br.replace_with("\n")
    for h in body.find_all(HEADINGS):
        h.replace_with(f"\n{HEADING_MARK}{' '.join(h.get_text(' ').split())}\n")
    for el in body.find_all(BLOCKS):
        el.insert_before("\n")
        el.insert_after("\n")
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for line in body.get_text(" ").split("\n"):
        line = " ".join(line.split())
        if line.startswith(HEADING_MARK):
            sections.append((line[len(HEADING_MARK) :].strip() or "section", []))
        elif line:
            sections[-1][1].append(line)
    return [(h, "\n\n".join(parts)) for h, parts in sections if parts]


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split hand-saved text at markdown headings ('#', '##', ...)."""
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for line in text.splitlines():
        m = re.match(r"#{1,4}\s+(.*)", line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines).strip()) for h, lines in sections if "\n".join(lines).strip()]


def pdf_sections(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return [
        (f"page-{i + 1}", page.extract_text() or "")
        for i, page in enumerate(reader.pages)
        if (page.extract_text() or "").strip()
    ]


def looks_like_pdf(content: bytes) -> bool:
    return content.lstrip()[:5] == b"%PDF-"


def extract_sections(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".pdf":
        head = path.read_bytes()[:1024]
        if not looks_like_pdf(head):
            if head.lstrip()[:1] == b"<":  # server sent a web page with a .pdf name
                return html_sections(path.read_text(encoding="utf-8", errors="replace"))
            raise ValueError("not a PDF file")
        return pdf_sections(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in (".html", ".htm"):
        return html_sections(text)
    return markdown_sections(text)


def build_scheme(entry: SourceEntry, records: list[ManifestEntry], raw_dir: Path) -> Scheme:
    """One Scheme per source entry. Section keys are 'd<doc index>-<heading slug>'."""
    sections: dict[str, str] = {}
    for d, record in enumerate(records):
        try:
            found = extract_sections(raw_dir / record.raw_path)
        except Exception as err:  # one broken file must not stop the whole run
            print(
                f"BROKEN {entry.scheme_id}: {record.raw_path} ({record.url}) could not be "
                f"read ({err.__class__.__name__}: {err}). It is probably a cut-off download. "
                f"Download it again in the browser, save it under data/manual/ with the name "
                f"`fetch` suggests, and rerun fetch and process."
            )
            found = []
        if not found:
            print(
                f"WARNING {entry.scheme_id}: no text in {record.raw_path} ({record.url}). "
                "The page may be built by JavaScript: save it from the browser after it "
                "loads, or use the scheme's guideline PDF."
            )
        for heading, text in found:
            key = f"d{d}-{slug(heading)}"
            n = 2
            while key in sections:
                key = f"d{d}-{slug(heading)}-{n}"
                n += 1
            sections[key] = text
    return Scheme(
        scheme_id=entry.scheme_id,
        name=entry.name,
        source_url=records[0].url,
        fetched_at=records[0].fetched_at,
        sections=sections,
    )


def process(entries: list[SourceEntry], raw_dir: Path, out_dir: Path, max_chars: int = 800) -> None:
    manifest = read_jsonl(raw_dir / "manifest.jsonl", ManifestEntry)
    schemes, passages = [], []
    for entry in entries:
        # latest record per URL, only for URLs still listed in sources.yaml, whose file exists
        current = {d.url for d in entry.docs}
        latest = {m.url: m for m in manifest if m.scheme_id == entry.scheme_id and m.url in current}
        records = []
        for m in latest.values():
            if (raw_dir / m.raw_path).exists():
                records.append(m)
            else:
                print(f"WARNING {entry.scheme_id}: {m.raw_path} is missing; run fetch again")
        if not records:
            print(f"skip {entry.scheme_id}: nothing fetched")
            continue
        scheme = build_scheme(entry, records, raw_dir)
        # each passage cites the URL of the document it came from
        url_by_doc = {f"d{d}": r.url for d, r in enumerate(records)}
        for p in chunk_scheme(scheme, max_chars):
            p.source_url = url_by_doc[p.section.split("-", 1)[0]]
            passages.append(p)
        schemes.append(scheme)
    write_jsonl(out_dir / "schemes.jsonl", schemes)
    write_jsonl(out_dir / "passages.jsonl", passages)
    print(f"{len(schemes)} schemes, {len(passages)} passages")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["fetch", "process"])
    ap.add_argument("--sources", type=Path, default=Path("data/sources.yaml"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, help="only the first N sources (pilot)")
    ap.add_argument("--only", nargs="+", metavar="SCHEME_ID", help="only these schemes")
    args = ap.parse_args()
    entries = load_sources(args.sources)
    if args.only:
        unknown = set(args.only) - {e.scheme_id for e in entries}
        if unknown:
            raise SystemExit(f"not in {args.sources}: {sorted(unknown)}")
        entries = [e for e in entries if e.scheme_id in args.only]
    entries = entries[: args.limit]
    if args.step == "fetch":
        new = fetch_all(entries, args.raw)
        print(f"fetched {len(new)} new documents")
    else:
        process(entries, args.raw, args.out)


if __name__ == "__main__":
    main()
