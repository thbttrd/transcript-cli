"""Configuration tree for the transcript pipeline.

Each sub-dataclass owns one stage's tunable parameters. The root `PipelineConfig`
threads through `pipeline.run()` and is the only object module-internal code reads
its hyperparameters from. Frozen so configs can be hashed/fingerprinted safely.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TranscribeConfig:
    model: str = "large-v3"
    language: str | None = None
    temperature: float = 0.0
    no_fallback: bool = True
    suppress_nst: bool = True


@dataclass(frozen=True)
class DiarizeConfig:
    streaming_preset: Literal["very_high_lat", "low_lat"] = "very_high_lat"
    num_speakers: int | None = None
    backend: Literal["sortformer", "diarizen"] = "sortformer"


@dataclass(frozen=True)
class AlignConfig:
    enabled: bool = True


@dataclass(frozen=True)
class MergeConfig:
    """Configures the word→speaker merge stage.

    `smooth_islands` toggles a post-pass that flips short same-speaker islands
    (size ≤ max_island_words) sandwiched between two runs of the same other
    speaker. Fixes the "y a" / "tes" boundary artefact where a connective word
    straddles a turn boundary and gets the wrong label.
    """
    smooth_islands: bool = True
    max_island_words: int = 2


@dataclass(frozen=True)
class LLMFixConfig:
    enabled: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    diarize:    DiarizeConfig    = field(default_factory=DiarizeConfig)
    align:      AlignConfig      = field(default_factory=AlignConfig)
    merge:      MergeConfig      = field(default_factory=MergeConfig)
    llm_fix:    LLMFixConfig     = field(default_factory=LLMFixConfig)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            transcribe=TranscribeConfig(**d["transcribe"]),
            diarize=DiarizeConfig(**d["diarize"]),
            align=AlignConfig(**d["align"]),
            merge=MergeConfig(**d["merge"]),
            llm_fix=LLMFixConfig(**d["llm_fix"]),
        )
