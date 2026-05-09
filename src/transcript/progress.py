import sys
import time
from typing import TextIO


class Progress:
    """Tiny stderr progress reporter — no rich/tqdm to keep deps minimal."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        quiet: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.stream = stream if stream is not None else sys.stderr
        self._step_started: dict[str, float] = {}

    def step(self, label: str) -> None:
        if self.quiet:
            return
        self._step_started[label] = time.monotonic()
        prefix = "→" if self.verbose else "·"
        self.stream.write(f"{prefix} {label}\n")
        self.stream.flush()

    def done(self, label: str) -> None:
        if self.quiet or not self.verbose:
            return
        elapsed = time.monotonic() - self._step_started.get(label, time.monotonic())
        self.stream.write(f"  ✓ {label} ok ({elapsed:.1f}s)\n")
        self.stream.flush()

    def warn(self, msg: str) -> None:
        if self.quiet:
            return
        self.stream.write(f"⚠️  {msg}\n")
        self.stream.flush()
