"""Persist per-(clip x config x tier) transcripts and word-level diffs.

Output paths:
  bench/results/transcripts/tier-N/<safe-clip-id>/<config_fingerprint>.json
  bench/results/diffs/tier-N/<safe-clip-id>/<config_fingerprint>.json

Both are JSON. Transcripts hold hypothesis + reference utterances; diffs are
INTENDED to hold meeteval-aligned word ops plus the cpWER speaker permutation,
but the meeteval per-row diff extraction is not yet implemented — `runner.py`
currently calls `save_diff` with empty `speaker_permutation={}` / `word_ops=[]`
placeholders so the on-disk layout exists. The persisted diff files therefore
contain `totals` of all zeros; do NOT use them as evidence until populated.
"""
import json
from dataclasses import asdict
from pathlib import Path

from transcript.models import Utterance


def _safe(clip_id: str) -> str:
    return clip_id.replace(":", "_").replace("/", "_")


def _path(results_dir: Path, kind: str, tier: int, clip_id: str, fp: str) -> Path:
    return results_dir / kind / f"tier-{tier}" / _safe(clip_id) / f"{fp}.json"


def save_transcript(
    *,
    results_dir: Path,
    tier: int,
    clip_id: str,
    config_fingerprint: str,
    hypothesis: list[Utterance],
    reference: list[Utterance],
) -> Path:
    path = _path(results_dir, "transcripts", tier, clip_id, config_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "config_fingerprint": config_fingerprint,
        "hypothesis": [asdict(u) for u in hypothesis],
        "reference":  [asdict(u) for u in reference],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def save_diff(
    *,
    results_dir: Path,
    tier: int,
    clip_id: str,
    config_fingerprint: str,
    speaker_permutation: dict[str, str],
    word_ops: list[dict],
) -> Path:
    path = _path(results_dir, "diffs", tier, clip_id, config_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"sub": 0, "ins": 0, "del": 0, "speaker_swap": 0}
    for op in word_ops:
        if op["op"] in totals:
            totals[op["op"]] += 1
    payload = {
        "clip_id": clip_id,
        "config_fingerprint": config_fingerprint,
        "speaker_permutation": speaker_permutation,
        "word_ops": word_ops,
        "totals": totals,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
