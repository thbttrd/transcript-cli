"""SUMM-RE loader.

SUMM-RE ships per-speaker tracks (3-4 speakers per meeting). To run the
transcript pipeline on a meeting we mix the tracks into a single 16 kHz mono
WAV with ffmpeg amix, and synthesise the reference RTTM/STM from the per-track
segment + word metadata.
"""
import random
import subprocess
import tempfile
from pathlib import Path

from bench.datasets.base import BenchClip, stm_line


class SUMMREDataset:
    name = "SUMM-RE"

    def __init__(self, *, cache_dir: Path):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "summ_re"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _iter_meetings(self):
        from datasets import load_dataset
        ds = load_dataset("linagora/SUMM-RE", split="dev", streaming=True)
        by_meeting: dict[str, list[dict]] = {}
        for row in ds:
            by_meeting.setdefault(row["meeting_id"], []).append(row)
        return by_meeting.items()

    def _prepare_clip(self, meeting_id: str, tracks: list[dict]) -> BenchClip | None:
        from transcript import audio as audio_mod
        wav_path = self.audio_dir / f"{meeting_id}.wav"
        rttm_path = wav_path.with_suffix(".rttm")
        stm_path  = wav_path.with_suffix(".stm")

        if not wav_path.exists():
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                track_paths = []
                import soundfile as sf
                for tr in tracks:
                    p = td_path / f"{tr['audio_id']}.wav"
                    sf.write(p, tr["audio"]["array"], tr["audio"]["sampling_rate"])
                    track_paths.append(p)
                _mix_tracks(track_paths, out_path=wav_path)

        if not rttm_path.exists():
            _synthesise_rttm(tracks, meeting_id=meeting_id, out_path=rttm_path)
        if not stm_path.exists():
            _synthesise_stm(tracks, meeting_id=meeting_id, out_path=stm_path)

        if not rttm_path.read_text().strip():
            return None
        n_speakers = len({tr["speaker_id"] for tr in tracks})
        if n_speakers > 4:
            return None

        duration = audio_mod._probe(wav_path)["duration"]
        return BenchClip(
            clip_id=f"SUMM-RE:{meeting_id}",
            audio_path=wav_path,
            language="fr",
            num_speakers=n_speakers,
            duration_s=duration,
            reference_rttm=rttm_path,
            reference_stm=stm_path,
        )

    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]:
        rng = random.Random(seed)
        meetings = list(self._iter_meetings())
        rng.shuffle(meetings)
        clips: list[BenchClip] = []
        for meeting_id, tracks in meetings:
            clip = self._prepare_clip(meeting_id, tracks)
            if clip is None:
                continue
            if max_duration_s is not None and clip.duration_s > max_duration_s:
                continue
            clips.append(clip)
            if len(clips) >= n:
                break
        return clips


def _mix_tracks(track_paths: list[Path], *, out_path: Path) -> None:
    """ffmpeg amix N tracks → 16 kHz mono PCM16 WAV."""
    n = len(track_paths)
    inputs: list[str] = []
    for p in track_paths:
        inputs += ["-i", str(p)]
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _synthesise_rttm(tracks: list[dict], *, meeting_id: str, out_path: Path) -> None:
    lines = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            start = float(seg["start"])
            dur = float(seg["end"]) - start
            lines.append(
                f"SPEAKER {meeting_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
            )
    out_path.write_text("\n".join(lines))


def _synthesise_stm(tracks: list[dict], *, meeting_id: str, out_path: Path) -> None:
    """One STM line per (speaker, sorted-by-start segment), text concatenated from words."""
    rows: list[tuple[float, float, str, str]] = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            words = seg.get("words") or []
            text = " ".join(w["word"] for w in words).strip()
            if not text:
                text = seg.get("transcript", "").strip()
            if not text:
                continue
            rows.append((float(seg["start"]), float(seg["end"]), spk, text))
    rows.sort()
    lines = [
        stm_line(meeting_id, spk, s, e, text)
        for s, e, spk, text in rows
    ]
    out_path.write_text("\n".join(lines))
