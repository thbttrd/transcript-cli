"""Shared types for bench datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchClip:
    """One audio clip ready for the bench harness.

    `audio_path` MUST be a 16 kHz mono PCM WAV. The runner skips `audio.prepare`
    on cached clips and feeds the path straight to transcribe/diarize/align —
    every dataset loader (`bench/datasets/ami.py`, `bench/datasets/summ_re.py`)
    is responsible for guaranteeing the format on its end.
    """
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


def stm_line(meeting_id: str, speaker: str, start: float, end: float, text: str) -> str:
    """Render one STM line. Shared by every dataset loader's reference-synthesis path."""
    return f"{meeting_id} 1 {speaker} {start:.2f} {end:.2f} <NA> {text}"
