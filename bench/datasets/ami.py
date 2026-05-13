"""AMI sdm corpus loader.

Reference RTTMs come from BUTSpeechFIT/AMI-diarization-setup, vendored under
bench/datasets/ami_rttm/. AMI audio is pulled from `edinburghcstr/ami` HF
dataset (sdm config — single distant mic). The HF dataset ships per-utterance
rows, so the loader splices each meeting's utterances into a single 16 kHz
mono WAV cached under bench/cache/audio/ami/.

If the vendored RTTM directory is empty on first run, the loader git-clones
the BUT repo into bench/cache/ami_rttm/ and descends into the nested
`only_words/rttms/test/` directory.
"""
import logging
import random
import subprocess
from pathlib import Path

from bench.datasets.base import BenchClip, stm_line

_log = logging.getLogger(__name__)

_HF_DATASET = "edinburghcstr/ami"
_HF_CONFIG  = "sdm"  # single distant mic
_HF_SPLIT   = "test"
_HF_SR      = 16000  # AMI sdm canonical rate
_BUT_REPO   = "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git"


def _load_dataset(*args, **kwargs):
    """Thin wrapper around ``datasets.load_dataset`` so tests can monkeypatch
    the call site without importing the heavy ``datasets`` package."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _build_meeting_wav(meeting_id: str, out_path: Path,
                       max_duration_s: float | None = None) -> None:
    """Splice all per-utterance rows for ``meeting_id`` from the HF AMI sdm
    dataset into one 16 kHz mono WAV at ``out_path``.

    Utterances are placed at the sample offset implied by their ``begin_time``.
    Overlap policy: first-writer-wins (earlier ``begin_time``). AMI sdm rows
    in overlap regions are slices of the same source mic, so summing would
    double the amplitude; first-writer-wins preserves the original level.

    If ``max_duration_s`` is set, the buffer is hard-clipped to that length;
    utterances that start after the cap are skipped.
    """
    import numpy as np
    import soundfile as sf

    ds = _load_dataset(_HF_DATASET, _HF_CONFIG, split=_HF_SPLIT, streaming=True)
    rows = [r for r in ds if r["meeting_id"] == meeting_id]
    if not rows:
        raise RuntimeError(f"no AMI rows found for meeting {meeting_id}")
    if max_duration_s is not None:
        rows = [r for r in rows if float(r["begin_time"]) < max_duration_s]
    rows.sort(key=lambda r: float(r["begin_time"]))

    max_end = (
        max_duration_s if max_duration_s is not None
        else max(float(r["end_time"]) for r in rows)
    )
    buf = np.zeros(int(max_end * _HF_SR) + _HF_SR, dtype=np.float32)

    for row in rows:
        src_sr = int(row["audio"]["sampling_rate"])
        if src_sr != _HF_SR:
            raise RuntimeError(
                f"unexpected sample rate {src_sr} for meeting {meeting_id} "
                f"(expected {_HF_SR}). AMI sdm rows should be 16 kHz."
            )
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        start = int(float(row["begin_time"]) * _HF_SR)
        end = start + len(arr)
        if end > len(buf):
            new_buf = np.zeros(end + _HF_SR, dtype=np.float32)
            new_buf[:len(buf)] = buf
            buf = new_buf
        mask = buf[start:end] == 0.0
        buf[start:end] = np.where(mask, arr, buf[start:end])

    if max_duration_s is not None:
        buf = buf[:int(max_duration_s * _HF_SR)]
    else:
        nonzero = np.nonzero(buf)[0]
        if len(nonzero) > 0:
            buf = buf[:nonzero[-1] + 1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, buf, _HF_SR, subtype="PCM_16")


def _truncate_rttm(src: Path, dst: Path, max_duration_s: float) -> None:
    """Copy ``src`` to ``dst``, dropping SPEAKER segments that start at or
    after ``max_duration_s`` and clipping segments that cross the boundary."""
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER" or len(parts) < 10:
            if line.strip():
                out_lines.append(line)
            continue
        start = float(parts[3])
        duration = float(parts[4])
        if start >= max_duration_s:
            continue
        end = start + duration
        if end > max_duration_s:
            parts[4] = f"{max_duration_s - start:.3f}"
        out_lines.append(" ".join(parts))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def _vendored_rttm_dir() -> Path:
    return Path(__file__).parent / "ami_rttm"


def _find_rttm_dir(root: Path, split: str = "test") -> Path | None:
    """Return the dir holding per-meeting RTTMs for ``split`` (default: test).

    The BUT repo layout is::

        <root>/only_words/rttms/
            train.rttm dev.rttm test.rttm     # split-level concatenated files
            train/<meeting_id>.rttm           # per-meeting RTTMs we want
            dev/<meeting_id>.rttm
            test/<meeting_id>.rttm

    The split-level *.rttm files at ``only_words/rttms/`` are NOT per-meeting
    and would mislead a shallow glob — descend into ``<split>/`` first.

    Falls back to a flat layout at ``only_words/rttms/`` or at ``root`` itself
    so manually-vendored RTTMs work too. Returns ``None`` if no per-meeting
    RTTMs are found anywhere.
    """
    nested = root / "only_words" / "rttms"
    split_dir = nested / split
    if split_dir.is_dir() and any(split_dir.glob("*.rttm")):
        return split_dir
    if nested.is_dir() and any(p for p in nested.glob("*.rttm")
                                if p.stem not in {"train", "dev", "test"}):
        return nested
    if any(root.glob("*.rttm")):
        return root
    return None


class AMIDataset:
    name = "AMI"

    def __init__(self, *, cache_dir: Path, rttm_dir: Path | None = None):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "ami"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.rttm_dir = rttm_dir or self._resolve_rttm_dir(cache_dir)
        self._index_cache: list[dict] | None = None
        self._stm_rows_by_meeting: dict[str, list[dict]] = {}

    @staticmethod
    def _resolve_rttm_dir(cache_dir: Path) -> Path:
        vendored = _vendored_rttm_dir()
        if vendored.exists() and any(vendored.glob("*.rttm")):
            return vendored
        runtime = cache_dir / "ami_rttm"
        if not runtime.exists():
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", _BUT_REPO, str(runtime)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else ""
                raise RuntimeError(
                    f"Could not fetch BUT RTTM repo from {_BUT_REPO}. "
                    f"Vendor RTTMs into {vendored} instead. Underlying: {stderr.strip()}"
                ) from e
        found = _find_rttm_dir(runtime)
        if found is None:
            raise RuntimeError(
                f"No RTTM files found under {runtime} (checked the BUT nested "
                f"layout only_words/rttms/ and the flat layout). Vendor RTTMs "
                f"into {vendored} or re-clone the BUT repo."
            )
        return found

    def _load_index(self) -> list[dict]:
        if self._index_cache is not None:
            return self._index_cache
        ds = _load_dataset(_HF_DATASET, _HF_CONFIG, split=_HF_SPLIT, streaming=True)
        seen: dict[str, dict] = {}
        stm_rows: dict[str, list[dict]] = {}
        for row in ds:
            mid = row["meeting_id"]
            end_t = float(row["end_time"])
            if mid not in seen:
                seen[mid] = {"meeting_id": mid, "duration": end_t}
            elif end_t > seen[mid]["duration"]:
                seen[mid]["duration"] = end_t
            stm_rows.setdefault(mid, []).append({
                "speaker_id": row.get("speaker_id", "Speaker_1"),
                "begin_time": float(row["begin_time"]),
                "end_time":   end_t,
                "text":       row["text"],
            })
        self._index_cache = list(seen.values())
        self._stm_rows_by_meeting = stm_rows
        return self._index_cache

    def _write_ami_stm(self, meeting_id: str, out: Path,
                        max_duration_s: float | None = None) -> None:
        """Write STM rows for one meeting from the cached HF index, optionally
        truncating segments past ``max_duration_s``."""
        if self._index_cache is None:
            self._load_index()
        lines: list[str] = []
        for r in self._stm_rows_by_meeting.get(meeting_id, []):
            begin = r["begin_time"]
            end = r["end_time"]
            if max_duration_s is not None:
                if begin >= max_duration_s:
                    continue
                end = min(end, max_duration_s)
            lines.append(stm_line(meeting_id, r["speaker_id"], begin, end, r["text"]))
        out.write_text("\n".join(lines))

    def _prepare_clip(self, meeting: dict,
                       max_duration_s: float | None = None) -> BenchClip | None:
        meeting_id = meeting["meeting_id"]
        full_rttm = self.rttm_dir / f"{meeting_id}.rttm"
        if not full_rttm.exists():
            _log.warning("AMI: skipping %s — no RTTM at %s", meeting_id, full_rttm)
            return None

        stem = _clip_stem(meeting_id, max_duration_s)
        wav_path = self.audio_dir / f"{stem}.wav"
        rttm_path = self.audio_dir / f"{stem}.rttm"
        stm_path = self.audio_dir / f"{stem}.stm"
        effective_duration = (
            min(meeting["duration"], max_duration_s) if max_duration_s
            else meeting["duration"]
        )

        if not wav_path.exists():
            _log.info("AMI: splicing %s into %s …", meeting_id, wav_path)
            _build_meeting_wav(meeting_id, wav_path, max_duration_s=max_duration_s)

        if not rttm_path.exists():
            if max_duration_s is None:
                rttm_path = full_rttm
            else:
                _truncate_rttm(full_rttm, rttm_path, max_duration_s)

        num_speakers = _count_rttm_speakers(rttm_path)
        if num_speakers > 4:
            _log.warning(
                "AMI: skipping %s — %d speakers exceeds Sortformer 4-speaker cap",
                meeting_id, num_speakers,
            )
            return None

        if not stm_path.exists():
            self._write_ami_stm(meeting_id, stm_path, max_duration_s=max_duration_s)

        return BenchClip(
            clip_id=f"AMI:{meeting_id}",
            audio_path=wav_path,
            language="en",
            num_speakers=num_speakers,
            duration_s=effective_duration,
            reference_rttm=rttm_path,
            reference_stm=stm_path,
        )

    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]:
        rng = random.Random(seed)
        clips: list[BenchClip] = []
        index = self._load_index()
        rng.shuffle(index)
        for meeting in index:
            clip = self._prepare_clip(meeting, max_duration_s=max_duration_s)
            if clip is not None:
                clips.append(clip)
            if len(clips) >= n:
                break
        return clips


def _clip_stem(meeting_id: str, max_duration_s: float | None) -> str:
    if max_duration_s is None:
        return meeting_id
    return f"{meeting_id}_{int(max_duration_s)}s"


def _count_rttm_speakers(rttm_path: Path) -> int:
    speakers = set()
    for line in rttm_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers)
