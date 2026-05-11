from pathlib import Path

from transcript.models import Turn

DIARIZER_LABEL = "NeMo Streaming Sortformer 4spk-v2.1"
_NEMO_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"

# Streaming Sortformer v2.1 processes audio in chunks of `chunk_len` frames
# (1 frame = 80 ms), so there's no fixed audio-length ceiling — the v1
# non-streaming path had a practical audio-length limit on consumer hardware.
#
# Two presets exist in NVIDIA's model card. We pick the "very high latency"
# one: this is a batch CLI, not a real-time stream, so we trade time-to-first-
# result (which we don't care about) for ~45× lower RTF and longer chunks
# (340 frames ≈ 27 s) that should give the model more context per decision.
#
#   |               | latency | RTF   | chunk | r-ctx | fifo | upd | cache |
#   | very-high-lat | 30.4 s  | 0.002 |  340  |  40   |  40  | 300 |  188  |
#   | low-latency   |  1.04 s | 0.093 |    6  |   7   | 188  | 144 |  188  |
_STREAMING_PARAMS = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    # NVIDIA's published preset is 300, but NeMo clamps the effective value to
    # at least chunk_len at runtime — passing 340 directly silences the
    # "less than chunk_len" warning without changing behavior.
    "spkcache_update_period": 340,
    "spkcache_len": 188,
}


_model = None


class DiarizeError(RuntimeError):
    """User-facing diarization error."""


def _load_model():
    """Lazy-load the Sortformer model once. Cached across calls within a process."""
    global _model
    if _model is not None:
        return _model
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as e:
        raise DiarizeError(
            "nemo_toolkit not installed. Re-run scripts/install.sh."
        ) from e
    # Stays on CPU: NeMo's MPS autocast path was unreliable last we checked,
    # and streaming chunks run well under real-time on CPU anyway.
    try:
        m = SortformerEncLabelModel.from_pretrained(_NEMO_MODEL, map_location="cpu")
    except Exception as e:
        raise DiarizeError(f"could not load NeMo Sortformer: {e}") from e
    m.train(False)
    for name, value in _STREAMING_PARAMS.items():
        setattr(m.sortformer_modules, name, value)
    m.sortformer_modules._check_streaming_parameters()
    _model = m
    return _model


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
    """Diarize `wav_path` with NeMo Streaming Sortformer; return Turns labeled Speaker 1..N."""
    model = _load_model()
    results = model.diarize(audio=[str(wav_path)], batch_size=1)
    raw_lines = results[0] if results else []
    raw = _parse_sortformer_segments(raw_lines)

    # Sortformer 4spk always emits up to 4 labels. If the user fixed --speakers
    # below 4, drop turns labeled beyond that count by first-appearance order.
    turns = _relabel(raw)
    if num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(num_speakers)}
        turns = [t for t in turns if t.speaker in keep]
    return turns
