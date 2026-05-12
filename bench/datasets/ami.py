"""AMI sdm corpus loader.

Reference RTTMs come from BUTSpeechFIT/AMI-diarization-setup, vendored under
bench/datasets/ami_rttm/. AMI audio is pulled from `edinburghcstr/ami` HF
dataset (sdm config — single distant mic) and pre-prepared into 16 kHz mono
WAVs cached under bench/cache/audio/ami/.

If the vendored RTTM directory is empty on first run, attempt to clone the
BUT repo into bench/cache/ami_rttm/ as a fallback (warns the user).
"""
import random
import shutil
import subprocess
from pathlib import Path

from bench.datasets.base import BenchClip

_HF_DATASET = "edinburghcstr/ami"
_HF_CONFIG  = "sdm"  # single distant mic
_BUT_REPO   = "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git"


class AMIDataset:
    name = "AMI"

    def __init__(self, *, cache_dir: Path, rttm_dir: Path | None = None):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "ami"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.rttm_dir = rttm_dir or self._resolve_rttm_dir(cache_dir)

    @staticmethod
    def _resolve_rttm_dir(cache_dir: Path) -> Path:
        vendored = Path(__file__).parent / "ami_rttm"
        if vendored.exists() and any(vendored.glob("*.rttm")):
            return vendored
        runtime = cache_dir / "ami_rttm"
        if not runtime.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", _BUT_REPO, str(runtime)],
                check=True,
            )
        # The BUT repo nests RTTMs under a subdirectory — adjust this path
        # after first clone if needed (open question in the spec).
        return runtime

    def _load_index(self) -> list[dict]:
        from datasets import load_dataset
        ds = load_dataset(_HF_DATASET, _HF_CONFIG, split="test")
        seen: dict[str, dict] = {}
        for row in ds:
            mid = row["meeting_id"]
            if mid not in seen:
                seen[mid] = {
                    "meeting_id": mid,
                    "audio": row["audio"]["path"],
                    "duration": float(row["audio"]["array"].shape[0])
                                / float(row["audio"]["sampling_rate"]),
                }
        return list(seen.values())

    def _prepare_clip(self, meeting: dict) -> BenchClip | None:
        from transcript import audio as audio_mod

        meeting_id = meeting["meeting_id"]
        rttm_file = self.rttm_dir / f"{meeting_id}.rttm"
        if not rttm_file.exists():
            return None

        wav_path = self.audio_dir / f"{meeting_id}.wav"
        if not wav_path.exists():
            prepared, _ = audio_mod.prepare(Path(meeting["audio"]))
            shutil.move(prepared, wav_path)

        num_speakers = _count_rttm_speakers(rttm_file)
        if num_speakers > 4:
            return None  # Sortformer 4-speaker cap

        stm_file = wav_path.with_suffix(".stm")
        if not stm_file.exists():
            _ami_stm_for(meeting_id, stm_file)

        return BenchClip(
            clip_id=f"AMI:{meeting_id}",
            audio_path=wav_path,
            language="en",
            num_speakers=num_speakers,
            duration_s=meeting["duration"],
            reference_rttm=rttm_file,
            reference_stm=stm_file,
        )

    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]:
        rng = random.Random(seed)
        clips: list[BenchClip] = []
        index = self._load_index()
        rng.shuffle(index)
        for meeting in index:
            if max_duration_s is not None and meeting["duration"] > max_duration_s:
                continue
            clip = self._prepare_clip(meeting)
            if clip is not None:
                clips.append(clip)
            if len(clips) >= n:
                break
        return clips


def _count_rttm_speakers(rttm_path: Path) -> int:
    speakers = set()
    for line in rttm_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers)


def _ami_stm_for(meeting_id: str, out: Path) -> None:
    """Build an STM file from AMI manual transcripts."""
    from datasets import load_dataset
    ds = load_dataset(_HF_DATASET, _HF_CONFIG, split="test")
    lines = []
    for row in ds:
        if row["meeting_id"] != meeting_id:
            continue
        speaker = row.get("speaker_id", "Speaker_1")
        begin = float(row["begin_time"])
        end = float(row["end_time"])
        text = row["text"]
        lines.append(
            f"{meeting_id} 1 {speaker} {begin:.2f} {end:.2f} <NA> {text}"
        )
    out.write_text("\n".join(lines))
