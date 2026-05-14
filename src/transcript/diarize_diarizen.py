"""DiariZen diarization backend, isolated via per-script uv env.

DiariZen (BUTSpeechFIT/DiariZen) requires pyannote.audio 4.x, which requires
torch>=2.7 — incompatible with this project's torch==2.6.0 pin (NeMo / whisper.cpp
constraint). To run both in one process: we don't. The actual DiariZen call lives
in `scripts/diarize_diarizen.py`, which carries its own PEP 723 inline deps and
runs under a separate uv-managed env. We invoke it as a subprocess and parse JSON
turns off its stdout. NC-licensed weights — personal use only.
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from transcript.models import Turn

if TYPE_CHECKING:
    from transcript.pipeline_config import DiarizeConfig

_log = logging.getLogger(__name__)

DIARIZER_LABEL = "DiariZen WavLM-Large s80-md (BUT-FIT)"

# Resolve scripts/diarize_diarizen.py relative to this file: project_root/scripts/...
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "diarize_diarizen.py"


class DiariZenError(RuntimeError):
    """User-facing DiariZen error."""


def _relabel_by_first_appearance(turns: list[Turn]) -> list[Turn]:
    """Renumber so 'Speaker 1' is whoever talks first, matching Sortformer's convention.

    DiariZen returns pyannote-style labels (e.g. "SPEAKER_00", "SPEAKER_01") assigned
    arbitrarily by its clustering step. The merge stage downstream and side-by-side
    comparison against Sortformer both rely on "Speaker N == Nth person to talk".
    """
    if not turns:
        return turns
    label_map: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t.start):
        if t.speaker not in label_map:
            label_map[t.speaker] = f"Speaker {len(label_map) + 1}"
    return [Turn(speaker=label_map[t.speaker], start=t.start, end=t.end) for t in turns]


def run(wav_path: Path, *, config: "DiarizeConfig") -> list[Turn]:
    """Diarize via the isolated DiariZen subprocess and return turns."""
    from transcript.pipeline_config import DiarizeConfig
    if not isinstance(config, DiarizeConfig):
        raise TypeError(f"config must be DiarizeConfig, got {type(config).__name__}")
    if not _SCRIPT.exists():
        raise DiariZenError(f"DiariZen runner script not found at {_SCRIPT}")

    try:
        proc = subprocess.run(
            ["uv", "run", "--script", str(_SCRIPT), str(wav_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise DiariZenError("`uv` not found on PATH — required for isolated DiariZen env") from e
    except subprocess.CalledProcessError as e:
        # First few hundred chars of stderr usually pinpoint the problem; truncate
        # so callers don't get a 50 KB traceback in their terminal.
        tail = (e.stderr or "")[-2000:]
        raise DiariZenError(
            f"DiariZen subprocess failed (exit {e.returncode}). stderr tail:\n{tail}"
        ) from e

    try:
        raw_turns = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise DiariZenError(
            f"DiariZen subprocess produced non-JSON stdout: {proc.stdout[:200]!r}"
        ) from e

    turns = [
        Turn(speaker=str(t["speaker"]), start=float(t["start"]), end=float(t["end"]))
        for t in raw_turns
    ]
    turns = _relabel_by_first_appearance(turns)

    if config.num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(config.num_speakers)}
        turns = [t for t in turns if t.speaker in keep]

    if not turns:
        _log.warning(
            "DiariZen returned no turns for %s — every word will be labelled Unknown",
            wav_path,
        )
    return turns
