import shutil

import torch

from transcript import config


def _nemo_importable() -> bool:
    try:
        import nemo.collections.asr  # noqa: F401
    except ImportError:
        return False
    return True


def _check(label: str, ok: bool, hint: str | None = None) -> tuple[bool, str]:
    mark = "✓" if ok else "✗"
    line = f"  {mark} {label}"
    if not ok and hint:
        line += f"\n      → {hint}"
    return ok, line


def check() -> tuple[int, str]:
    """Run all boundary checks; return (exit_code, multiline_report)."""
    results: list[tuple[bool, str]] = []

    bin_path = config.whisper_binary()
    results.append(_check(
        f"whisper.cpp binary at {bin_path}",
        bin_path.exists(),
        hint="run scripts/install.sh to clone and build whisper.cpp",
    ))

    model_path = config.whisper_model("large-v3")
    results.append(_check(
        f"whisper model at {model_path}",
        model_path.exists(),
        hint="run scripts/install.sh to download ggml-large-v3.bin",
    ))

    encoder_path = config.whisper_coreml_encoder("large-v3")
    results.append(_check(
        f"CoreML encoder at {encoder_path}",
        encoder_path.exists(),
        hint="run scripts/install.sh to generate the CoreML encoder",
    ))

    results.append(_check(
        "ffmpeg on $PATH",
        shutil.which("ffmpeg") is not None,
        hint="brew install ffmpeg",
    ))

    results.append(_check(
        "nemo_toolkit importable",
        _nemo_importable(),
        hint="re-run scripts/install.sh",
    ))

    results.append(_check(
        "PyTorch MPS available",
        torch.backends.mps.is_available(),
        hint="diarization will fall back to CPU (slower)",
    ))

    report_lines = ["transcript --doctor"] + [line for _, line in results]
    code = 0 if all(ok for ok, _ in results) else 1
    return code, "\n".join(report_lines)
