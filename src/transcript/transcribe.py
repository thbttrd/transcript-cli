import json
import subprocess
import tempfile
from pathlib import Path

from transcript import config
from transcript.models import Word


class TranscribeError(RuntimeError):
    """User-facing transcription error."""


def _parse_words(data: dict) -> list[Word]:
    """Pull word-level tokens out of whisper.cpp's --output-json-full payload."""
    words: list[Word] = []
    for segment in data.get("transcription", []):
        for tok in segment.get("tokens", []):
            text: str = tok.get("text", "")
            stripped = text.strip()
            # Skip whisper's special marker tokens like [_BEG_], [_TT_3], etc.
            if not stripped or stripped.startswith("[_"):
                continue
            offsets = tok.get("offsets", {})
            start_ms = int(offsets.get("from", 0))
            end_ms = int(offsets.get("to", 0))
            words.append(Word(text=text, start=start_ms / 1000.0, end=end_ms / 1000.0))
    return words


def run(wav_path: Path, *, model: str, language: str | None) -> list[Word]:
    """Transcribe a 16 kHz mono WAV using whisper.cpp; return word-level Words."""
    binary = config.whisper_binary()
    if not binary.exists():
        raise TranscribeError(
            f"whisper.cpp binary not found at {binary}. Run scripts/install.sh."
        )
    model_path = config.whisper_model(model)
    if not model_path.exists():
        raise TranscribeError(
            f"whisper model {model_path.name} not found. Run scripts/install.sh."
        )

    out_prefix = Path(tempfile.mkdtemp(prefix="transcript-")) / "whisper-out"
    cmd = [
        str(binary),
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", language or "auto",
        "-ml", "1",
        "--split-on-word",
        "-ojf",
        "-of", str(out_prefix),
        "--no-prints",
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise TranscribeError(f"whisper.cpp failed: {stderr.strip()}") from e

    json_file = out_prefix.with_suffix(out_prefix.suffix + ".json")
    if not json_file.exists():
        # whisper.cpp writes <prefix>.json
        json_file = Path(str(out_prefix) + ".json")
    data = json.loads(json_file.read_text())
    return _parse_words(data)
