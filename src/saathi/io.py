"""JSONL helpers for Pydantic records."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def append_jsonl(path: Path, record: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
