"""Shared types for bench datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchClip:
    clip_id: str
    audio_path: Path
    language: str       # ISO 639-1
    num_speakers: int
    duration_s: float
    reference_rttm: Path
    reference_stm: Path


class Dataset(Protocol):
    name: str

    def sample(
        self,
        n: int,
        *,
        max_duration_s: float | None = None,
        seed: int = 42,
    ) -> list[BenchClip]: ...
