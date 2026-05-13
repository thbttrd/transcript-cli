# TODO

Ideas to test, ordered by expected payoff.

## 1. Probability-based per-word speaker assignment — DONE → REVERTED (§4-ter)

Originally shipped as `merge.strategy = "prob_based"`: Sortformer emitted a
[T × 4] per-frame probability tensor when `DiarizeConfig.emit_probs=True`
and `merge._best_speaker_prob_based` argmaxed over each word's frame
window. The Tier 1+2 bench (see §4) showed the strategy was 33–67 pp
worse than hard-boundary on cpWER, so the entire code path was reverted
in §4-ter. Keeping the section here so the chain-of-reasoning stays
visible from §1 → §4 → §4-ter.

## 2. Sortformer v2.1 — DONE

Shipped: `nvidia/diar_streaming_sortformer_4spk-v2.1` via NeMo 2.7.3, configured
with the "very-high-latency" preset (chunk_len=340, RTF≈0.002). Removes the v1
~12-minute ceiling — streaming chunks handle arbitrarily long audio.

The blocker described here originally (transformers ≤4.48.3 vs ≥4.50) turned
out to be illusory: transformers 4.57.6 (pulled in by NeMo 2.7.3) accepts
both `dtype=` and the legacy `torch_dtype=`, so `align.py`'s manual model
load needed only a one-character kwarg switch.

## 3. Quantitative benchmarking on real datasets — DONE

Shipped: `scripts/benchmark.py` runs a two-tier search on AMI (ihm split,
mixed into a single 16 kHz mono WAV per meeting) and SUMM-RE (dev split,
ffmpeg amix of per-speaker tracks) using cpWER (via meeteval) as the
primary metric, with WER, DER, and a "speaker-assignment error rate"
(cpWER - WER) as secondary diagnostics. Results are appended to
`bench/results/runs.csv` and auto-summarised in
`bench/results/leaderboard.md`. Per-row hypothesis and diff artefacts
land under `bench/results/{transcripts,diffs}/tier-N/` for post-hoc
failure-mode analysis.

## 4. Run the benchmark and pick a winning config — DONE

Tier 1 and Tier 2 ran on 2026-05-13 against AMI ihm-mixed and SUMM-RE dev,
producing 352 rows of evidence (now archived at `bench/results/runs.v1.csv`
under the pre-revert schema with the `merge_strategy` column). Tier 3 was
deliberately skipped: the winner was unambiguous and running 50 clips ×
full-duration AMI would only amplify the whisper hallucination noise
documented under "Follow-ups" below. Tier 3 has since been removed
entirely (§4-ter).

### Verdict — the default config was the bench winner up to noise

Tier 1 ranked the **32 configs** (the old `merge_strategy` axis added
2× to the 16-config grid the harness now runs) by median cpWER. The
pre-bench defaults landed at rank #3 with median cpWER 69.08%, only
**0.39 percentage points** behind the rank-#1 config (cpWER 68.69%).
After dropping the `merge_strategy` axis and flipping `no_fallback`
to the bench-best value, the production default is now byte-for-byte
the rank-#1 config (§4-ter).

| Axis | Pre-bench default | Bench-best in Tier 1 | Δ | Notes |
|---|---|---|---|---|
| `transcribe.no_fallback` | `True` | `False` | 0.39 pp | **Flipped to False** in §4-ter to match the bench winner. |
| `transcribe.suppress_nst` | `True` | `True` | 0 | Default already optimal. |
| `diarize.streaming_preset` | `very_high_lat` | `very_high_lat` | 0 | `low_lat` is 0.3–0.5 pp worse on hard_boundary. Default already optimal. |
| `align.enabled` | `True` | `True` | 0 | `False` is within 1 pp; align contributes marginal value at this clip length. |
| `merge.strategy` | `hard_boundary` | `hard_boundary` | 0 | `prob_based` was 33–67 pp worse — reverted entirely in §4-ter. |

### Follow-up: whisper hallucination on long AMI clips

AMI Tier 2 (900s clips) revealed catastrophic whisper hallucination on
specific meetings (TS3003a 747% WER, TS3003c 790%, ES2004c 519%). Root cause
diagnosed: the per-utterance AMI ihm splice writes **hard zeros** in regions
where no speaker has an utterance row, which triggers whisper-large-v3's
known hallucination loop on artificial silence + `condition_on_previous_text`.
SUMM-RE (ffmpeg amix of full per-speaker tracks) does **not** show this
because each track carries continuous room tone, so silent regions have a
natural noise floor.

Mitigations (not blocking the verdict above — the rankings hold regardless):
- Inject low-level pink/white noise at ≈ −50 dB into the silent regions of
  `_build_meeting_wav`.
- Set whisper `condition_on_previous_text=False` in `transcribe.py` (the
  loop-amplifying setting).
- Or both. Track under a new TODO when this becomes a real bottleneck.

### Artefacts on disk

- `bench/results/runs.v1.csv` — 352 rows (192 Tier 1 + 160 Tier 2) under
  the pre-revert schema. Frozen historical evidence; the live bench
  writes a fresh `runs.csv` on next run.
- `bench/results/leaderboard.v1.md` — Tier 2 leaderboard per dataset,
  rendered from the v1 CSV. Frozen.
- `bench/results/transcripts/tier-{1,2}/...` — per-(clip × config)
  hypothesis + reference utterances. Gitignored; useful for the
  follow-up failure-mode analysis.

---

## 4-bis. Original Tier-3 runbook — removed

Previously this section walked through running `scripts/benchmark.py
--tier 3` (50 finalists × full-duration clips). With `prob_based` reverted
(§4-ter) the only race Tier 3 settled is gone, the CLI no longer accepts
`--tier 3`, and the runbook would only mislead. The original instructions
are still recoverable from git history (`git log -- docs/todo.md`) if
needed for context.

## 4-ter. Revert `merge.strategy = "prob_based"` — DONE

Tier 1+2 showed `prob_based` was consistently 33–67 pp **worse** than
`hard_boundary` on cpWER. The strategy never earned its keep; the
[T×4] probability tensor it depended on added latency, memory, and code
surface for negative ROI. Production-default (`hard_boundary`) was
also the pre-bench default, so the revert was behaviour-preserving on
the production CLI path.

Shipped:
- Removed `MergeConfig` dataclass, `DiarizeConfig.emit_probs`, and the
  prob-based code path from `merge.py`, `diarize.py`, and `pipeline.py`.
  `diarize.run` now returns `list[Turn]` directly.
- Dropped the `merge_strategy` axis from `bench/tiers.py` (Tier 1 grid
  shrank from 32 to 16 configs) and the `merge_strategy` CSV column
  from `bench/runner.py`. Existing `bench/results/runs.csv` was renamed
  to `runs.v1.csv` (likewise `leaderboard.md` → `leaderboard.v1.md`) so
  a fresh bench run starts from a clean slate without clobbering the
  historical evidence. A header-mismatch guard in `run_one_tier`
  prevents future silent schema drift.
- Deleted Tier 3 entirely. Without the `prob_based`-vs-`hard_boundary`
  race, Tier 3 would have collapsed to "re-run defaults on more clips"
  — and the verdict already showed defaults capture ~99.4% of
  achievable gain.
- Flipped `TranscribeConfig.no_fallback` default to `False` to match
  the Tier 1 rank-#1 config (Δ 0.39 pp, within clip noise but the
  evidence points one way).
- Pruned `bench/cache.py` (emit_probs gone from sortformer_key; probs
  npy artefact gone). Added a schema-version field to the sortformer
  cache JSON so caches written under the pre-revert code are detected
  and re-run rather than silently reused.
- Updated/removed all affected tests; full non-integration suite green
  (165 tests).

---

## 5. Populate per-row meeteval diffs — FOLLOW-UP

`bench/runner.py:run_one_tier` currently calls `artefacts.save_diff` with
empty placeholders (`speaker_permutation={}`, `word_ops=[]`). The persisted
diff JSON files therefore have all-zero `totals`. To make them useful for
failure-mode analysis (see #6), wire `meeteval`'s alignment output into
`runner.py:_run_cached` → `metrics.score` → `artefacts.save_diff`:

- meeteval's `cpwer()` result exposes `assignment` (the speaker permutation)
  and a per-word alignment when called with SegLST inputs.
- The diff schema in `artefacts.py` already expects `op ∈ {equal, sub, ins,
  del, speaker_swap}`. The `speaker_swap` op (right word, wrong speaker) is
  exactly the failure mode the prob_based merge attempted to target.

## 6. LLM-fix prompt-design tailored to residual failures — FOLLOW-UP

Deliberately excluded from the benchmark axes (per the project's
"tune deterministic core first" decision). The deterministic config is
now frozen (§4-ter); the next step is to mine
`bench/results/diffs/tier-2/` for residual error patterns (boundary
swaps, stranded short islands, hallucinated inserts) and rewrite the
`llm_fix.apply` prompt to target only what the deterministic pipeline
can't fix. Without this, `llm_fix`'s gemma4:e4b keeps introducing more
errors than it removes on cases the deterministic pipeline already
handles. Gated on #5 — the diff JSONs are currently all-zero.
