"""SUMM-RE loader.

SUMM-RE ships per-speaker tracks (3-4 speakers per meeting). To run the
transcript pipeline on a meeting we mix the tracks into a single 16 kHz mono
WAV with ffmpeg amix, and synthesise the reference RTTM/STM from the per-track
segment + word metadata.

Memory model: the dev split is streamed once, with the ``audio`` column cast
to ``Audio(decode=False)`` so HF returns raw file bytes instead of decoded
numpy arrays. Bytes are written straight to disk for ffmpeg to mix — neither
all the per-track buffers nor a full-split row dict ever live in RAM at the
same time. Rows are processed meeting-by-meeting; only the meeting currently
being assembled is held in memory.

Selection: meetings are taken in stream order (deterministic but not random).
The ``seed`` arg is accepted for API parity with other dataset loaders but is
currently unused — HF streaming doesn't give cheap random access, and the
buffered-shuffle alternative breaks the per-meeting contiguity this loader
relies on. Cache the meetings you want to pin and the same set will be picked
up on warm runs.
"""
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from bench.datasets.base import BenchClip, stm_line

_log = logging.getLogger(__name__)

_HF_DATASET = "linagora/SUMM-RE"
_HF_SPLIT = "dev"
# Matches the duration suffix added by _prepare_clip — e.g. "_60s", "_900s".
# Used to distinguish full-length cache entries from truncated ones.
_DURATION_SUFFIX_RE = re.compile(r"_\d+s$")


def _load_dataset(*args, **kwargs):
    """Thin wrapper so tests can monkeypatch without importing ``datasets``."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _cast_audio_no_decode(ds):
    """Cast the ``audio`` column to ``Audio(decode=False)`` if supported.

    With decoding off the column yields ``{"path": ..., "bytes": ...}`` for
    parquet-embedded audio (SUMM-RE's case), letting us skip the float32
    numpy materialisation entirely.
    """
    try:
        from datasets import Audio
    except ImportError:
        return ds
    if hasattr(ds, "cast_column"):
        return ds.cast_column("audio", Audio(decode=False))
    return ds


class SUMMREDataset:
    name = "SUMM-RE"

    def __init__(self, *, cache_dir: Path):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "summ_re"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _warn_orphan_clips(self) -> None:
        """Surface WAVs without their .rttm/.stm siblings — almost always the
        result of an interrupted prior streaming run. Silently re-streaming
        each time would waste bandwidth; warning lets the user clean up."""
        for wav_path in sorted(self.audio_dir.glob("*.wav")):
            missing = [
                ext for ext in ("rttm", "stm")
                if not wav_path.with_suffix(f".{ext}").exists()
            ]
            if missing:
                _log.warning(
                    "SUMM-RE: orphan cache entry %s (missing .%s) — likely an "
                    "interrupted previous run; rm to re-stream or ignore",
                    wav_path, ", .".join(missing),
                )

    def _cached_clips(self, max_duration_s: float | None,
                       limit: int) -> list[BenchClip]:
        """Return up to ``limit`` BenchClips reconstructed from on-disk cache.

        Lets warm runs skip the streaming pass entirely. A meeting is considered
        cached iff its wav, rttm and stm all exist with non-empty RTTM.
        """
        from transcript import audio as audio_mod

        suffix = "" if max_duration_s is None else f"_{int(max_duration_s)}s"
        pattern = f"*{suffix}.wav" if suffix else "*.wav"

        out: list[BenchClip] = []
        for wav_path in sorted(self.audio_dir.glob(pattern)):
            stem = wav_path.stem
            if suffix and not stem.endswith(suffix):
                continue
            meeting_id = stem[:-len(suffix)] if suffix else stem
            if not meeting_id:
                continue
            # When asked for full-length clips, skip stems that carry a
            # duration suffix (e.g. "<id>_60s.wav") so we don't mistake a
            # truncated cache entry for the real meeting.
            if not suffix and _DURATION_SUFFIX_RE.search(meeting_id):
                continue
            rttm_path = wav_path.with_suffix(".rttm")
            stm_path  = wav_path.with_suffix(".stm")
            if not (rttm_path.exists() and stm_path.exists()):
                continue
            rttm_text = rttm_path.read_text()
            if not rttm_text.strip():
                continue
            speakers = {
                line.split()[7]
                for line in rttm_text.splitlines()
                if line.startswith("SPEAKER") and len(line.split()) >= 8
            }
            if not speakers or len(speakers) > 4:
                continue
            out.append(BenchClip(
                clip_id=f"SUMM-RE:{meeting_id}",
                audio_path=wav_path,
                language="fr",
                num_speakers=len(speakers),
                duration_s=audio_mod._probe(wav_path)["duration"],
                reference_rttm=rttm_path,
                reference_stm=stm_path,
            ))
            if len(out) >= limit:
                break
        return out

    def _prepare_clip(self, meeting_id: str, tracks: list[dict],
                       max_duration_s: float | None = None) -> BenchClip | None:
        from transcript import audio as audio_mod
        stem = meeting_id if max_duration_s is None else f"{meeting_id}_{int(max_duration_s)}s"
        wav_path = self.audio_dir / f"{stem}.wav"
        rttm_path = wav_path.with_suffix(".rttm")
        stm_path  = wav_path.with_suffix(".stm")

        if not wav_path.exists():
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                track_paths = [
                    _write_track(td_path, tr) for tr in tracks
                ]
                _mix_tracks(track_paths, out_path=wav_path, max_duration_s=max_duration_s)

        if not rttm_path.exists():
            _synthesise_rttm(tracks, meeting_id=meeting_id, out_path=rttm_path,
                             max_duration_s=max_duration_s)
        if not stm_path.exists():
            _synthesise_stm(tracks, meeting_id=meeting_id, out_path=stm_path,
                            max_duration_s=max_duration_s)

        if not rttm_path.read_text().strip():
            _log.warning("SUMM-RE: skipping %s — synthesised RTTM is empty", meeting_id)
            return None
        n_speakers = len({tr["speaker_id"] for tr in tracks})
        if n_speakers > 4:
            _log.warning(
                "SUMM-RE: skipping %s — %d speakers exceeds Sortformer 4-speaker cap",
                meeting_id, n_speakers,
            )
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
        """Return up to ``n`` BenchClips, streaming the dev split if needed.

        Fast path: glob the cache for already-mixed meetings and return those
        directly. Slow path: stream the dev split with audio decode disabled,
        flush each meeting as soon as its rows finish, stop when we hit ``n``.

        Peak RAM is bounded to the raw bytes of one meeting's per-speaker
        tracks (~30-60 MB for a 30-minute meeting), not the whole split.
        """
        del seed  # see module docstring — currently unused.
        self._warn_orphan_clips()
        cached = self._cached_clips(max_duration_s, limit=n)
        if len(cached) >= n:
            _log.info("SUMM-RE: %d cached clip(s) cover the request, skipping stream", n)
            return cached[:n]

        ds = _load_dataset(_HF_DATASET, split=_HF_SPLIT, streaming=True)
        ds = _cast_audio_no_decode(ds)

        # Reuse cached entries as the starting list, but drop them from the
        # streaming side via a skip-set so we don't redo work.
        clips = list(cached)
        skip_meeting_ids = {c.clip_id.split(":", 1)[1] for c in cached}

        current_id: str | None = None
        buffer: list[dict] = []
        _log.info("SUMM-RE: streaming dev split (need %d more clip(s))", n - len(clips))

        def flush() -> bool:
            """Process buffered tracks for ``current_id``. Returns True if we
            should keep streaming, False if we have enough."""
            nonlocal buffer
            if current_id is not None and current_id not in skip_meeting_ids and buffer:
                clip = self._prepare_clip(current_id, buffer,
                                           max_duration_s=max_duration_s)
                if clip is not None:
                    clips.append(clip)
                    _log.info("SUMM-RE: mixed %s (%d/%d)",
                              current_id, len(clips), n)
            buffer = []  # drop bytes refs
            return len(clips) < n

        for row in ds:
            mid = row["meeting_id"]
            if mid != current_id:
                if not flush():
                    break
                current_id = mid
            if mid in skip_meeting_ids:
                continue
            buffer.append(row)
        else:
            flush()

        return clips[:n]


def _write_track(td_path: Path, tr: dict) -> Path:
    """Write a single per-speaker track from an HF row to ``td_path``.

    Handles both decoded mode (``audio.array`` + ``audio.sampling_rate``) and
    undecoded mode (``audio.bytes`` + ``audio.path``). Undecoded is the hot
    path — we cast the column to ``Audio(decode=False)`` so HF hands us the
    raw file bytes, which we drop straight onto disk for ffmpeg to ingest.
    """
    audio = tr["audio"]
    audio_id = tr.get("audio_id") or audio.get("path", "track")
    if isinstance(audio, dict) and audio.get("bytes") is not None:
        src_path = audio.get("path") or f"{audio_id}.wav"
        ext = Path(src_path).suffix or ".wav"
        p = td_path / f"{Path(audio_id).stem}{ext}"
        p.write_bytes(audio["bytes"])
        return p
    import soundfile as sf
    p = td_path / f"{Path(audio_id).stem}.wav"
    sf.write(p, audio["array"], audio["sampling_rate"])
    return p


def _mix_tracks(track_paths: list[Path], *, out_path: Path,
                max_duration_s: float | None = None) -> None:
    """ffmpeg amix N tracks → 16 kHz mono PCM16 WAV. If ``max_duration_s`` is
    set, the output is clipped to that length via ffmpeg ``-t``."""
    n = len(track_paths)
    inputs: list[str] = []
    for p in track_paths:
        inputs += ["-i", str(p)]
    duration_args = ["-t", str(max_duration_s)] if max_duration_s else []
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        *duration_args,
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg amix failed: {stderr.strip()}") from e


def _synthesise_rttm(tracks: list[dict], *, meeting_id: str, out_path: Path,
                      max_duration_s: float | None = None) -> None:
    lines = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            start = float(seg["start"])
            end = float(seg["end"])
            if max_duration_s is not None:
                if start >= max_duration_s:
                    continue
                end = min(end, max_duration_s)
            dur = end - start
            lines.append(
                f"SPEAKER {meeting_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
            )
    out_path.write_text("\n".join(lines))


def _synthesise_stm(tracks: list[dict], *, meeting_id: str, out_path: Path,
                     max_duration_s: float | None = None) -> None:
    """One STM line per (speaker, sorted-by-start segment), text concatenated from words."""
    rows: list[tuple[float, float, str, str]] = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            start = float(seg["start"])
            end = float(seg["end"])
            if max_duration_s is not None:
                if start >= max_duration_s:
                    continue
                end = min(end, max_duration_s)
            words = seg.get("words") or []
            text = " ".join(w["word"] for w in words).strip()
            if not text:
                text = seg.get("transcript", "").strip()
            if not text:
                continue
            rows.append((start, end, spk, text))
    rows.sort()
    lines = [
        stm_line(meeting_id, spk, s, e, text)
        for s, e, spk, text in rows
    ]
    out_path.write_text("\n".join(lines))
