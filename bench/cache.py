"""Content-hashed on-disk cache for the three expensive pipeline stages.

Each cached artefact's key incorporates a sha1 of the input audio plus only
the config fields that affect that stage's output. Serialisation: JSON for
structured data, NumPy `.npy` for tensors — no opaque binary formats.
"""
import functools
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from transcript.models import Turn, Word
from transcript.pipeline_config import DiarizeConfig, TranscribeConfig


@functools.cache
def _audio_sha1_by_resolved(resolved: str) -> str:
    h = hashlib.sha1()
    with open(resolved, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_sha1(audio_path: Path) -> str:
    return _audio_sha1_by_resolved(str(audio_path.resolve()))


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def whisper_key(audio_path: Path, cfg: TranscribeConfig) -> str:
    return _hash(
        audio_sha1(audio_path),
        "whisper",
        json.dumps(asdict(cfg), sort_keys=True),
    )


def sortformer_key(audio_path: Path, cfg: DiarizeConfig) -> str:
    # num_speakers is a post-hoc filter in diarize.run, not a model input,
    # so it's safe to exclude from the cache key — same audio + preset always
    # yields the same pre-filter turns regardless of num_speakers.
    relevant = {
        "streaming_preset": cfg.streaming_preset,
        "emit_probs": cfg.emit_probs,
    }
    return _hash(
        audio_sha1(audio_path),
        "sortformer",
        json.dumps(relevant, sort_keys=True),
    )


def align_key(audio_path: Path, whisper_hash: str, language: str) -> str:
    return _hash(audio_sha1(audio_path), "align", whisper_hash, language)


def _whisper_path(audio_path: Path, cfg: TranscribeConfig, cache_dir: Path) -> Path:
    return cache_dir / "whisper" / f"{whisper_key(audio_path, cfg)}.json"


def save_whisper(
    audio_path: Path,
    cfg: TranscribeConfig,
    words: list[Word],
    *,
    cache_dir: Path,
) -> None:
    path = _whisper_path(audio_path, cfg, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"text": w.text, "start": w.start, "end": w.end} for w in words])
    )


def load_whisper(
    audio_path: Path,
    cfg: TranscribeConfig,
    *,
    cache_dir: Path,
) -> list[Word] | None:
    path = _whisper_path(audio_path, cfg, cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [Word(**r) for r in raw]


def _sortformer_dir(audio_path: Path, cfg: DiarizeConfig, cache_dir: Path) -> Path:
    return cache_dir / "sortformer" / sortformer_key(audio_path, cfg)


def save_sortformer(
    audio_path: Path,
    cfg: DiarizeConfig,
    turns: list[Turn],
    *,
    probs: np.ndarray | None,
    cache_dir: Path,
) -> None:
    base = _sortformer_dir(audio_path, cfg, cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "turns.json").write_text(
        json.dumps(
            [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns]
        )
    )
    if probs is not None:
        np.save(base / "probs.npy", probs)


def load_sortformer(
    audio_path: Path,
    cfg: DiarizeConfig,
    *,
    cache_dir: Path,
) -> tuple[list[Turn], np.ndarray | None] | None:
    base = _sortformer_dir(audio_path, cfg, cache_dir)
    turns_file = base / "turns.json"
    if not turns_file.exists():
        return None
    turns = [Turn(**r) for r in json.loads(turns_file.read_text())]
    probs_file = base / "probs.npy"
    probs = np.load(probs_file) if probs_file.exists() else None
    return turns, probs


def _align_path(
    audio_path: Path, whisper_hash: str, language: str, cache_dir: Path
) -> Path:
    return cache_dir / "align" / f"{align_key(audio_path, whisper_hash, language)}.json"


def save_align(
    audio_path: Path,
    whisper_hash: str,
    language: str,
    words: list[Word],
    *,
    cache_dir: Path,
) -> None:
    path = _align_path(audio_path, whisper_hash, language, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"text": w.text, "start": w.start, "end": w.end} for w in words])
    )


def load_align(
    audio_path: Path,
    whisper_hash: str,
    language: str,
    *,
    cache_dir: Path,
) -> list[Word] | None:
    path = _align_path(audio_path, whisper_hash, language, cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [Word(**r) for r in raw]
