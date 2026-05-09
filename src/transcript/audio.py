import json
import subprocess
import tempfile
from pathlib import Path

MIN_DURATION_S = 0.5


class AudioError(RuntimeError):
    """User-facing audio preparation error."""


def _probe(path: Path) -> dict:
    """Return basic metadata for `path` via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except FileNotFoundError as e:
        raise AudioError("ffprobe not found — install ffmpeg (`brew install ffmpeg`)") from e
    except subprocess.CalledProcessError as e:
        raise AudioError(f"could not read audio: {e.stderr.strip()}") from e

    data = json.loads(result.stdout)
    audio_streams = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if not audio_streams:
        raise AudioError(f"no audio stream in {path}")
    s = audio_streams[0]
    return {
        "sample_rate": int(s["sample_rate"]),
        "channels": int(s["channels"]),
        "duration": float(data["format"]["duration"]),
    }


def prepare(path: Path) -> tuple[Path, float]:
    """Prepare audio for whisper.cpp.

    Returns (wav_path, duration_seconds).
    Passes through if already 16 kHz mono WAV; otherwise converts via ffmpeg
    to a temp WAV that the caller is expected to clean up later
    (we leave it for OS-level temp cleanup since pyannote may also need it).
    """
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")

    info = _probe(path)
    duration = info["duration"]
    if duration < MIN_DURATION_S:
        raise AudioError(f"audio too short to transcribe ({duration:.2f}s)")

    is_correct_wav = (
        path.suffix.lower() == ".wav"
        and info["sample_rate"] == 16000
        and info["channels"] == 1
    )
    if is_correct_wav:
        return path, duration

    out_path = Path(tempfile.mkstemp(suffix=".wav", prefix="transcript-")[1])
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", str(path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise AudioError("ffmpeg not found — install with `brew install ffmpeg`") from e
    except subprocess.CalledProcessError as e:
        raise AudioError(f"ffmpeg conversion failed: {e.stderr.decode().strip()}") from e

    return out_path, duration
