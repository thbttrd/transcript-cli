import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from transcript import config as transcript_config
from transcript.models import Word

if TYPE_CHECKING:
    from transcript.pipeline_config import TranscribeConfig


class TranscribeError(RuntimeError):
    """User-facing transcription error."""


def _parse_words(data: dict) -> list[Word]:
    """Pull word-level segments out of whisper.cpp's --output-json-full payload.

    Whisper.cpp runs with `--max-len 1 --split-on-word`, so each *segment* is one
    word — possibly several BPE tokens long ("Chouchou" → "Ch" + "ouch" + "ou").
    Consume segment text directly; iterating per-token would let diarization
    scatter the pieces of a single word across different speakers.
    """
    words: list[Word] = []
    for segment in data.get("transcription", []):
        text: str = segment.get("text", "")
        stripped = text.strip()
        # Skip whisper's special marker segments like [_BEG_], [_TT_3], etc.
        if not stripped or stripped.startswith("[_"):
            continue
        offsets = segment.get("offsets", {})
        start_ms = int(offsets.get("from", 0))
        end_ms = int(offsets.get("to", 0))
        words.append(Word(text=text, start=start_ms / 1000.0, end=end_ms / 1000.0))
    return words


def _detected_language(data: dict, fallback: str | None) -> str:
    """Pull the language whisper.cpp actually used (after auto-detect if applicable)."""
    return data.get("result", {}).get("language") or fallback or "auto"


def run(wav_path: Path, *, config: "TranscribeConfig") -> tuple[list[Word], str]:
    """Transcribe a 16 kHz mono WAV using whisper.cpp."""
    from transcript.pipeline_config import TranscribeConfig
    assert isinstance(config, TranscribeConfig)

    binary = transcript_config.whisper_binary()
    if not binary.exists():
        raise TranscribeError(
            f"whisper.cpp binary not found at {binary}. Run scripts/install.sh."
        )
    model_path = transcript_config.whisper_model(config.model)
    if not model_path.exists():
        raise TranscribeError(
            f"whisper model {model_path.name} not found. Run scripts/install.sh."
        )

    with tempfile.TemporaryDirectory(prefix="transcript-") as tmpdir:
        out_prefix = Path(tmpdir) / "whisper-out"
        cmd: list[str] = [
            str(binary),
            "-m", str(model_path),
            "-f", str(wav_path),
            "-l", config.language or "auto",
            "-ml", "1",
            "--split-on-word",
            "--temperature", str(config.temperature),
            "-ojf",
            "-of", str(out_prefix),
            "--no-prints",
        ]
        if config.no_fallback:
            cmd.append("--no-fallback")
        if config.suppress_nst:
            cmd.append("--suppress-nst")

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise TranscribeError(f"whisper.cpp failed: {stderr.strip()}") from e

        json_file = Path(str(out_prefix) + ".json")
        data = json.loads(json_file.read_text())
        return _parse_words(data), _detected_language(data, config.language)
