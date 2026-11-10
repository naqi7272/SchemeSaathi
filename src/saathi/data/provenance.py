"""Provenance of question data: license gate and the dataset manifest.

Every external or generated question set is recorded in data/qa/manifest.jsonl: name,
license, source, how it was made (human-written, synthetic, scraped, machine-made), which
model made it (if any), and counts. Datasets with unclear licenses or scraped from
myscheme.gov.in are refused, never silently used.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MANIFEST = Path("data/qa/manifest.jsonl")

Provenance = Literal["human", "synthetic", "scraped", "machine_translit", "machine_translation"]

# SPDX-style ids (lower-case) that allow reuse in a public research benchmark.
OPEN_LICENSES = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "odc-by",
    "odbl",
    "pddl",
    "cdla-permissive-2.0",
    "cdla-sharing-1.0",
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "gpl-3.0",
    "lgpl-3.0",
    "agpl-3.0",
    "ogdl-india",  # Government Open Data License - India
    "project-own",  # written for this project by its author(s)
}
# Non-commercial licenses are usable for a research benchmark but must be flagged.
NONCOMMERCIAL_LICENSES = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-3.0", "cc-by-nc-sa-3.0"}
BLOCKED_SOURCES = ("myscheme.gov.in",)


class LicenseError(ValueError):
    pass


def check_license(license_id: str, source_url: str, provenance: str) -> list[str]:
    """Raise LicenseError if the data must not be used; return warnings otherwise."""
    lic = (license_id or "").strip().lower()
    blocked = [b for b in BLOCKED_SOURCES if b in (source_url or "").lower()]
    if blocked:
        raise LicenseError(f"source {source_url!r} is {blocked[0]}, whose terms forbid bots")
    if not lic or lic in {"unknown", "other", "unclear", "none"}:
        raise LicenseError(
            f"license {license_id!r} is unclear; check the dataset card and the original "
            "source, then set an explicit license id"
        )
    warnings = []
    if lic in NONCOMMERCIAL_LICENSES:
        warnings.append(f"license {lic} is non-commercial: fine for research, record it")
    elif lic not in OPEN_LICENSES:
        raise LicenseError(
            f"license {license_id!r} is not in the known open list; check it by hand and add "
            "it to OPEN_LICENSES in saathi/data/provenance.py only if reuse is allowed"
        )
    if provenance == "scraped":
        warnings.append(
            "dataset is scraped: confirm the ORIGINAL site's terms allow it (not only the "
            "dataset's own license)"
        )
    return warnings


class ManifestRecord(BaseModel):
    name: str
    kind: Literal["open_dataset", "handwritten", "machine_variant", "llm_generated"]
    license: str
    source: str  # URL, HF id, or file path
    provenance: Provenance
    generator_model: str | None = None
    generator_family_warning: str | None = None
    out_file: str
    n_rows_in: int
    n_items_out: int
    n_dropped: int
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    config_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(rec: ManifestRecord, path: Path = MANIFEST) -> None:
    """Append (never edit) one manifest record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")


def now() -> datetime:
    return datetime.now(UTC)
