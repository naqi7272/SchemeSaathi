"""The default path must need no API key: guard the shipped configs and defaults."""

from pathlib import Path

import pytest
import yaml

from saathi.eval.run import RunConfig

RUN_CONFIGS = sorted(Path("configs/runs").glob("*.yaml"))


def test_there_are_default_configs():
    assert len(RUN_CONFIGS) >= 9


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_default_run_configs_are_free_and_valid(path):
    cfg = RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert cfg.backend != "claude", f"{path}: paid configs belong in configs/optional-paid/"
    assert not cfg.model.startswith("claude")


def test_optional_paid_configs_are_clearly_marked():
    for path in Path("configs/optional-paid").glob("*.yaml"):
        assert path.read_text(encoding="utf-8").startswith("# OPTIONAL, PAID")


def test_anthropic_is_not_a_core_dependency():
    import tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(d.startswith("anthropic") for d in project["dependencies"])
    assert any(d.startswith("anthropic") for d in project["optional-dependencies"]["claude"])
