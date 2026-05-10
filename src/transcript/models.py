from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass(frozen=True)
class Turn:
    speaker: str  # e.g. "Speaker 1"
    start: float
    end: float


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Meta:
    filename: str
    duration: float           # seconds
    model: str                # e.g. "large-v3"
    language: str             # ISO code e.g. "fr"
    speaker_count: int        # 1 if --no-diarize, otherwise diarizer-detected count
    diarizer: str | None = None  # human-readable backend label, None when --no-diarize
