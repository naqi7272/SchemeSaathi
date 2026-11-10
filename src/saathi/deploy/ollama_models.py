"""Register GGUF files from the fine-tuning notebook as Ollama models.

Expects files named <model>.<QUANT>.gguf (e.g. qwen3-1.7b-saathi.Q8_0.gguf) and creates
Ollama models named <model>:<quant> (e.g. qwen3-1.7b-saathi:q8_0), which the eval configs use.
The chat template and end-of-turn token come from the GGUF metadata; the stop token is also
set explicitly. Sampling settings are not baked in; the eval client sends them per call.

Usage:
    uv run python -m saathi.deploy.ollama_models gguf/
    uv run python -m saathi.deploy.ollama_models gguf/ --dry-run
"""

import argparse
import re
import subprocess
from pathlib import Path

NAME = re.compile(r"(?P<model>[a-z0-9.-]+)\.(?P<quant>F16|Q8_0|Q4_K_M)\.gguf", re.IGNORECASE)


def modelfile(gguf: Path) -> str:
    return f'FROM {gguf.resolve().as_posix()}\nPARAMETER stop "<|im_end|>"\n'


def plan(folder: Path) -> list[tuple[str, Path]]:
    """(ollama model name, gguf path) for every recognised file, sorted by name."""
    out = []
    for path in sorted(folder.glob("*.gguf")):
        m = NAME.fullmatch(path.name)
        if m:
            out.append((f"{m['model'].lower()}:{m['quant'].lower()}", path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for name, path in plan(args.folder):
        mf = path.with_suffix(".Modelfile")
        mf.write_text(modelfile(path), encoding="utf-8")
        print(f"{name}  <-  {path.name}")
        if not args.dry_run:
            subprocess.run(["ollama", "create", name, "-f", str(mf)], check=True)


if __name__ == "__main__":
    main()
