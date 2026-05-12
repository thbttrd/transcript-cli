from pathlib import Path
from typing import TYPE_CHECKING

from transcript.models import Turn

if TYPE_CHECKING:
    from transcript.pipeline_config import DiarizeConfig

DIARIZER_LABEL = "NeMo Streaming Sortformer 4spk-v2.1"
_NEMO_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"

# Streaming Sortformer v2.1 processes audio in chunks of `chunk_len` frames
# (1 frame = 80 ms), so there's no fixed audio-length ceiling — the v1
# non-streaming path had a practical audio-length limit on consumer hardware.
#
# Two presets exist in NVIDIA's model card:
#
#   |               | latency | RTF   | chunk | r-ctx | fifo | upd | cache |
#   | very-high-lat | 30.4 s  | 0.002 |  340  |  40   |  40  | 300 |  188  |
#   | low-latency   |  1.04 s | 0.093 |    6  |   7   | 188  | 144 |  188  |
_STREAMING_PRESETS: dict[str, dict[str, int]] = {
    "very_high_lat": {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        # NVIDIA's published preset is 300, but NeMo clamps the effective value
        # to at least chunk_len at runtime — passing 340 directly silences the
        # "less than chunk_len" warning without changing behavior.
        "spkcache_update_period": 340,
        "spkcache_len": 188,
    },
    "low_lat": {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
}


class DiarizeError(RuntimeError):
    """User-facing diarization error."""


def _streaming_params(preset: str) -> dict[str, int]:
    if preset not in _STREAMING_PRESETS:
        raise DiarizeError(f"unknown streaming preset: {preset}")
    return dict(_STREAMING_PRESETS[preset])


_model_cache: dict[str, object] = {}


def _load_model(preset: str = "very_high_lat"):
    """Lazy-load Sortformer once per preset. Cached across calls within a process."""
    if preset in _model_cache:
        return _model_cache[preset]
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
    for name, value in _streaming_params(preset).items():
        setattr(m.sortformer_modules, name, value)
    m.sortformer_modules._check_streaming_parameters()
    _model_cache[preset] = m
    return m


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


def run(wav_path: Path, *, config: "DiarizeConfig") -> tuple[list[Turn], "np.ndarray | None"]:
    """Diarize and return (turns, optional [T x 4] probability tensor)."""
    from transcript.pipeline_config import DiarizeConfig
    assert isinstance(config, DiarizeConfig)

    model = _load_model(config.streaming_preset)
    if config.emit_probs:
        result = model.diarize(
            audio=[str(wav_path)], batch_size=1, include_tensor_outputs=True
        )
        # NeMo returns (segments_list, tensor_list) when include_tensor_outputs=True.
        if isinstance(result, tuple) and len(result) == 2:
            raw_lines = result[0][0] if result[0] else []
            probs = result[1] if not isinstance(result[1], list) else result[1][0]
        else:
            raw_lines = result[0] if result else []
            probs = None
    else:
        results = model.diarize(audio=[str(wav_path)], batch_size=1)
        raw_lines = results[0] if results else []
        probs = None

    raw = _parse_sortformer_segments(raw_lines)
    turns = _relabel(raw)
    if config.num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(config.num_speakers)}
        turns = [t for t in turns if t.speaker in keep]
    return turns, probs
