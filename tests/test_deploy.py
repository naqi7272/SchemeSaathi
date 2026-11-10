from saathi.deploy.ollama_models import modelfile, plan


def test_plan_names_models_and_skips_unknown_files(tmp_path):
    for name in [
        "qwen3-1.7b-saathi.Q8_0.gguf",
        "qwen3-1.7b-base.Q4_K_M.gguf",
        "qwen3-1.7b-base.F16.gguf",
        "random.gguf",
        "notes.txt",
    ]:
        (tmp_path / name).write_bytes(b"")
    assert [n for n, _ in plan(tmp_path)] == [
        "qwen3-1.7b-base:f16",
        "qwen3-1.7b-base:q4_k_m",
        "qwen3-1.7b-saathi:q8_0",
    ]


def test_modelfile_points_at_absolute_path(tmp_path):
    gguf = tmp_path / "m.Q8_0.gguf"
    text = modelfile(gguf)
    assert text.startswith(f"FROM {gguf.resolve().as_posix()}\n")
    assert 'PARAMETER stop "<|im_end|>"' in text
