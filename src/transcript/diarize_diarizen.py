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

from transcript import diarize_common
from transcript.models import Turn

if TYPE_CHECKING:
    from transcript.pipeline_config import DiarizeConfig

_log = logging.getLogger(__name__)

DIARIZER_LABEL = "DiariZen WavLM-Large s80-md (BUT-FIT)"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "diarize_diarizen.py"


class DiariZenError(RuntimeError):
    """User-facing DiariZen error."""


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
    turns = diarize_common.relabel_by_first_appearance(turns)
    return diarize_common.filter_and_warn(
        turns,
        num_speakers=config.num_speakers,
        backend_label="DiariZen",
        wav_path=wav_path,
        log=_log,
    )
