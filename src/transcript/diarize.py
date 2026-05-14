import logging
from pathlib import Path
from typing import TYPE_CHECKING

from transcript import diarize_common
from transcript.diarize_common import DiarizeError  # re-export
from transcript.models import Turn

__all__ = ["DIARIZER_LABEL", "DiarizeError", "run"]

if TYPE_CHECKING:
    from transcript.pipeline_config import DiarizeConfig

_log = logging.getLogger(__name__)

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


def run(wav_path: Path, *, config: "DiarizeConfig") -> list[Turn]:
    """Diarize and return the list of speaker turns."""
    from transcript.pipeline_config import DiarizeConfig
    if not isinstance(config, DiarizeConfig):
        raise TypeError(f"config must be DiarizeConfig, got {type(config).__name__}")

    model = _load_model(config.streaming_preset)
    results = model.diarize(audio=[str(wav_path)], batch_size=1)
    raw_lines = results[0] if results else []

    raw = _parse_sortformer_segments(raw_lines)
    turns = [Turn(speaker=label, start=s, end=e) for s, e, label in raw]
    turns = diarize_common.relabel_by_first_appearance(turns)
    return diarize_common.filter_and_warn(
        turns,
        num_speakers=config.num_speakers,
        backend_label="Sortformer",
        wav_path=wav_path,
        log=_log,
    )
