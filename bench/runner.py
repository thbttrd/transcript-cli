"""Per-tier execution: for each (clip x config) pair, run the cached pipeline,
score the result, append a CSV row, and persist transcripts + diffs.
"""
import csv
import logging
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

from bench import artefacts, cache, metrics
from bench.datasets.base import BenchClip, Dataset
from transcript import align as align_mod
from transcript import diarize, merge, transcribe
from transcript.models import Meta, Utterance
from transcript.pipeline_config import PipelineConfig

_log = logging.getLogger(__name__)

CACHED_STAGE_S = -1.0  # CSV sentinel: stage hit the cache, no measurement taken.

CSV_COLUMNS = [
    "tier", "dataset", "clip_id", "config_id", "config_fingerprint",
    "no_fallback", "suppress_nst", "streaming_preset", "align",
    "cpwer", "wer", "der", "speaker_assignment_error_rate",
    "runtime_s", "whisper_s", "sortformer_s", "align_s", "merge_s",
    "git_sha", "started_at", "host",
    "hypothesis_path", "diff_path",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_reference_utterances(stm_path: Path) -> list[Utterance]:
    """Parse a synthesised STM into Utterance objects."""
    out: list[Utterance] = []
    for lineno, line in enumerate(stm_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=6)
        if len(parts) < 7:
            _log.warning(
                "STM %s line %d: skipping malformed row with %d cols (need 7): %r",
                stm_path, lineno, len(parts), line,
            )
            continue
        _file, _ch, speaker, start, end, _na, text = parts
        out.append(Utterance(
            speaker=speaker, start=float(start), end=float(end), text=text
        ))
    return out


def _run_cached(clip: BenchClip, cfg: PipelineConfig,
                cache_dir: Path) -> tuple[list[Utterance], Meta, dict]:
    """Run the pipeline for one (clip x config), reading from / writing to the cache."""
    timings = {
        "whisper_s":    CACHED_STAGE_S,
        "sortformer_s": CACHED_STAGE_S,
        "align_s":      CACHED_STAGE_S,
        "merge_s":      CACHED_STAGE_S,
    }

    transcribe_cfg = replace(cfg.transcribe, language=clip.language)
    diarize_cfg = replace(cfg.diarize, num_speakers=clip.num_speakers)

    words = cache.load_whisper(clip.audio_path, transcribe_cfg, cache_dir=cache_dir)
    if words is None:
        t = time.time()
        words, _lang = transcribe.run(clip.audio_path, config=transcribe_cfg)
        timings["whisper_s"] = time.time() - t
        cache.save_whisper(clip.audio_path, transcribe_cfg, words, cache_dir=cache_dir)

    turns = cache.load_sortformer(clip.audio_path, diarize_cfg, cache_dir=cache_dir)
    if turns is None:
        t = time.time()
        turns = diarize.run(clip.audio_path, config=diarize_cfg)
        timings["sortformer_s"] = time.time() - t
        cache.save_sortformer(clip.audio_path, diarize_cfg, turns, cache_dir=cache_dir)

    if cfg.align.enabled and align_mod.is_available() and words:
        whisper_h = cache.whisper_key(clip.audio_path, transcribe_cfg)
        aligned = cache.load_align(
            clip.audio_path, whisper_h, clip.language, cache_dir=cache_dir
        )
        if aligned is None:
            t = time.time()
            aligned = align_mod.run(clip.audio_path, words, language=clip.language)
            timings["align_s"] = time.time() - t
            cache.save_align(
                clip.audio_path, whisper_h, clip.language, aligned, cache_dir=cache_dir
            )
        words = aligned

    t = time.time()
    word_speakers = merge.assign_speakers(words, turns)
    utterances = merge.collapse(word_speakers)
    timings["merge_s"] = time.time() - t

    meta = Meta(
        filename=clip.audio_path.name,
        duration=clip.duration_s,
        model=cfg.transcribe.model,
        language=clip.language,
        speaker_count=len({t.speaker for t in turns}) if turns else 0,
        diarizer=diarize.DIARIZER_LABEL,
    )
    return utterances, meta, timings


def run_one_tier(
    *,
    tier: int,
    configs: list[PipelineConfig],
    datasets: list[Dataset],
    clip_count: int,
    max_duration_s: float | None,
    cache_dir: Path,
    results_dir: Path,
) -> None:
    """Execute every (clip x config) pair for one tier; append to runs.csv."""
    git_sha = _git_sha()
    host = socket.gethostname()
    csv_path = results_dir / "runs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()

        for dataset in datasets:
            clips = dataset.sample(clip_count, max_duration_s=max_duration_s)
            ref_by_clip = {c.clip_id: _load_reference_utterances(c.reference_stm) for c in clips}
            for clip in clips:
                for cfg in configs:
                    fp = cfg.fingerprint()
                    started = time.time()
                    utterances, _meta, timings = _run_cached(clip, cfg, cache_dir)
                    m = metrics.score(utterances, ref_by_clip[clip.clip_id])

                    hyp_path = artefacts.save_transcript(
                        results_dir=results_dir, tier=tier,
                        clip_id=clip.clip_id, config_fingerprint=fp,
                        hypothesis=utterances, reference=ref_by_clip[clip.clip_id],
                    )
                    diff_path = artefacts.save_diff(
                        results_dir=results_dir, tier=tier,
                        clip_id=clip.clip_id, config_fingerprint=fp,
                        speaker_permutation={},
                        word_ops=[],
                    )

                    writer.writerow({
                        "tier": tier, "dataset": dataset.name,
                        "clip_id": clip.clip_id, "config_id": fp,
                        "config_fingerprint": fp,
                        "no_fallback": cfg.transcribe.no_fallback,
                        "suppress_nst": cfg.transcribe.suppress_nst,
                        "streaming_preset": cfg.diarize.streaming_preset,
                        "align": cfg.align.enabled,
                        "cpwer": m.cpwer, "wer": m.wer, "der": m.der,
                        "speaker_assignment_error_rate": m.speaker_assignment_error_rate,
                        "runtime_s": time.time() - started,
                        **timings,
                        "git_sha": git_sha,
                        "started_at": started, "host": host,
                        "hypothesis_path": str(hyp_path.relative_to(results_dir)),
                        "diff_path": str(diff_path.relative_to(results_dir)),
                    })
                    f.flush()


def generate_leaderboard(*, results_dir: Path) -> Path:
    """Rebuild leaderboard.md from runs.csv. Median cpWER per (dataset x config)
    using the highest tier present in runs.csv — so the file is meaningful even
    when only Tier 1 has run."""
    csv_path = results_dir / "runs.csv"
    out_path = results_dir / "leaderboard.md"
    if not csv_path.exists():
        out_path.write_text("# Benchmark leaderboard\n\n_No runs yet._\n")
        return out_path

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        out_path.write_text("# Benchmark leaderboard\n\n_No runs yet._\n")
        return out_path

    top_tier = max(int(r["tier"]) for r in rows)
    tier_rows = [r for r in rows if r["tier"] == str(top_tier)]

    lines = [f"# Benchmark leaderboard (tier {top_tier}, median)\n"]
    for dataset in sorted({r["dataset"] for r in tier_rows}):
        lines.append(f"\n## {dataset}\n")
        lines.append("| Rank | Config | cpWER | WER | DER | Speaker-err | Runtime |")
        lines.append("|------|--------|-------|-----|-----|-------------|---------|")
        agg: dict[tuple, list[tuple[float, ...]]] = {}
        for r in tier_rows:
            if r["dataset"] != dataset:
                continue
            key = (r["no_fallback"], r["suppress_nst"], r["streaming_preset"],
                   r["align"])
            agg.setdefault(key, []).append((
                float(r["cpwer"]), float(r["wer"]), float(r["der"]),
                float(r["speaker_assignment_error_rate"]), float(r["runtime_s"]),
            ))
        ranked = sorted(
            ((k, median(c for c, *_ in v),
              median(w for _, w, *_ in v),
              median(d for _, _, d, *_ in v),
              median(s for _, _, _, s, _ in v),
              median(rt for *_, rt in v))
             for k, v in agg.items()),
            key=lambda x: x[1],
        )
        for rank, (k, c, w, d, s, rt) in enumerate(ranked, 1):
            nf, sn, sp, al = k
            label = (
                f"align={al}, sortformer={sp}, "
                f"no_fallback={nf}, suppress_nst={sn}"
            )
            row = (
                f"| {rank} | {label} | {c*100:.1f} | {w*100:.1f} | "
                f"{d*100:.1f} | {s*100:.1f} | {rt:.1f}s |"
            )
            lines.append(row)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
