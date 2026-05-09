from pathlib import Path

import torch
from pyannote.audio import Pipeline

from transcript import config
from transcript.models import Turn


class DiarizeError(RuntimeError):
    """User-facing diarization error."""


_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"


def _to_turns(annotation) -> list[Turn]:
    """Relabel pyannote's speaker IDs as Speaker 1, Speaker 2, … in first-appearance order."""
    label_map: dict[str, str] = {}
    turns: list[Turn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = f"Speaker {len(label_map) + 1}"
        turns.append(Turn(speaker=label_map[label], start=segment.start, end=segment.end))
    return turns


def run(wav_path: Path, *, num_speakers: int | None) -> list[Turn]:
    """Diarize `wav_path` using pyannote 3.1; return turns relabeled as Speaker N."""
    token = config.hf_token()
    try:
        pipeline = Pipeline.from_pretrained(_PIPELINE_NAME, use_auth_token=token)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise DiarizeError(
                f"pyannote refused the download (HTTP {status}). "
                "Likely cause: license not accepted.\n"
                "  1. Sign in at https://huggingface.co\n"
                "  2. Click 'Agree' on:\n"
                "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "     - https://huggingface.co/pyannote/segmentation-3.0\n"
                "  3. Re-run the same command."
            ) from e
        raise DiarizeError(f"could not load pyannote pipeline: {e}") from e

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline.to(torch.device(device))

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["min_speakers"] = num_speakers
        kwargs["max_speakers"] = num_speakers
    annotation = pipeline(str(wav_path), **kwargs)
    return _to_turns(annotation)
