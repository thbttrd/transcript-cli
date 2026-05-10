from pathlib import Path

from transcript.models import Turn

DIARIZER_LABEL = "NeMo Sortformer 4spk-v1"
_NEMO_MODEL = "nvidia/diar_sortformer_4spk-v1"


class DiarizeError(RuntimeError):
    """User-facing diarization error."""


def _relabel(raw_labels: list[tuple[float, float, str]]) -> list[Turn]:
    """Convert raw (start, end, label) tuples into Turns with stable Speaker N labels."""
    label_map: dict[str, str] = {}
    turns: list[Turn] = []
    for start, end, label in raw_labels:
        if label not in label_map:
            label_map[label] = f"Speaker {len(label_map) + 1}"
        turns.append(Turn(speaker=label_map[label], start=start, end=end))
    return turns


def _parse_sortformer_segments(segments: list[str]) -> list[tuple[float, float, str]]:
    """Parse Sortformer outputs ("start end speaker_id") into tuples."""
    out: list[tuple[float, float, str]] = []
    for line in segments:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        out.append((start, end, parts[2]))
    return out


def run(wav_path: Path, *, num_speakers: int | None) -> list[Turn]:
    """Diarize `wav_path` with NeMo Sortformer; return Turns labeled Speaker 1..N."""
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as e:
        raise DiarizeError(
            "nemo_toolkit not installed. Re-run scripts/install.sh."
        ) from e

    # NeMo's MPS autocast path is broken in 2.2.1; CPU runs in seconds anyway.
    try:
        model = SortformerEncLabelModel.from_pretrained(_NEMO_MODEL, map_location="cpu")
    except Exception as e:
        raise DiarizeError(f"could not load NeMo Sortformer: {e}") from e
    model.train(False)

    results = model.diarize(audio=[str(wav_path)], batch_size=1)
    raw_lines = results[0] if results else []
    raw = _parse_sortformer_segments(raw_lines)

    # Sortformer 4spk-v1 always emits up to 4 labels. If the user fixed --speakers
    # below 4, drop turns labeled beyond that count by first-appearance order.
    turns = _relabel(raw)
    if num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(num_speakers)}
        turns = [t for t in turns if t.speaker in keep]
    return turns
