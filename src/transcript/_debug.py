"""Tempdir-based diagnostic logging shared across optional pipeline features.

Each feature gets a stable per-process log file under `$TMPDIR/transcript-<name>.log`
so users can grep for failures without re-running with -v. All writes swallow
OSError so a borked tempdir never crashes the pipeline.
"""
import os
import tempfile


def log_path(name: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"transcript-{name}.log")


def write(path: str, text: str, *, append: bool = False) -> None:
    try:
        with open(path, "a" if append else "w") as f:
            f.write(text)
    except OSError:
        pass
