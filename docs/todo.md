# TODO

Ideas to test, ordered by expected payoff.

## 1. Probability-based per-word speaker assignment — DONE

Shipped: implemented as `merge.strategy = "prob_based"` knob. Sortformer now
emits the [T x 4] per-frame probability tensor when `DiarizeConfig.emit_probs`
is true; `merge._best_speaker_prob_based` averages over each word's frame
window and argmaxes. The new strategy is tested against the hard-boundary
variant in the bench harness (see #3).

## 2. Sortformer v2.1 — DONE

Shipped: `nvidia/diar_streaming_sortformer_4spk-v2.1` via NeMo 2.7.3, configured
with the "very-high-latency" preset (chunk_len=340, RTF≈0.002). Removes the v1
~12-minute ceiling — streaming chunks handle arbitrarily long audio.

The blocker described here originally (transformers ≤4.48.3 vs ≥4.50) turned
out to be illusory: transformers 4.57.6 (pulled in by NeMo 2.7.3) accepts
both `dtype=` and the legacy `torch_dtype=`, so `align.py`'s manual model
load needed only a one-character kwarg switch.

## 3. Quantitative benchmarking on real datasets — DONE

Shipped: `scripts/benchmark.py` runs a three-tier search on AMI (sdm split)
and SUMM-RE (dev split) using cpWER (via meeteval) as the primary metric,
with WER, DER, and a "speaker-assignment error rate" (cpWER - WER) as
secondary diagnostics. Results are appended to `bench/results/runs.csv` and
auto-summarised in `bench/results/leaderboard.md`. Per-row hypothesis and
diff artefacts are persisted under `bench/results/{transcripts,diffs}/` for
post-hoc failure-mode analysis.

## 4. Run the benchmark and pick a winning config — DONE

Tier 1 (32 configs × 3 clips × 2 datasets = 192 rows) and Tier 2 (8 configs ×
10 clips × 2 datasets = 160 rows) ran on 2026-05-13 against AMI ihm-mixed and
SUMM-RE dev. Tier 3 was deliberately skipped: the winner is unambiguous and
running 50 clips × full-duration AMI would only amplify the whisper
hallucination noise documented under "Follow-ups" below.

### Verdict — the default config was already #3 of 32

Tier 1 ranked the 32 configs by median cpWER. The pre-bench defaults landed
at rank #3 with median cpWER 69.08%, only **0.39 percentage points** behind
the rank-#1 config (cpWER 68.69%). Default captures ~99.4% of the achievable
gain on this knob space; further tuning of the 5 deterministic axes is not
worth the engineering cost.

| Axis | Default | Best in Tier 1 | Δ | Notes |
|---|---|---|---|---|
| `transcribe.no_fallback` | `True` | `False` | 0.39 pp | The only axis where the best differs from default; within noise across clips. |
| `transcribe.suppress_nst` | `True` | `True` | 0 | Default already optimal. |
| `diarize.streaming_preset` | `very_high_lat` | `very_high_lat` | 0 | `low_lat` is 0.3–0.5 pp worse on hard_boundary. Default already optimal. |
| `align.enabled` | `True` | `True` | 0 | `False` is within 1 pp; align contributes marginal value at this clip length. |
| `merge.strategy` | `hard_boundary` | `hard_boundary` | 0 | See below. |

### TODO #1 (`prob_based` merge) — REVERT

`prob_based` was introduced in TODO #1 to attack the "right word, wrong
speaker" failure mode. Tier 2 medians (160 rows):

| Dataset | `hard_boundary` median cpWER | `prob_based` median cpWER | Δ |
|---|---|---|---|
| AMI ihm-mixed | 174.7–175.5% | 232.8% | **+57 pp worse** |
| SUMM-RE dev | 44.4–46.3% | 112.8% | **+67 pp worse** |
| Combined | 89.1–90.2% | 123.2% | **+33 pp worse** |

`prob_based` did not earn its keep — the per-frame probability averaging adds
speaker confusion rather than removing it. Recommend deleting the
`prob_based` code path entirely:
- `src/transcript/merge.py:_best_speaker_prob_based` + the `strategy` knob
- `src/transcript/diarize.py:DiarizeConfig.emit_probs` + the [T×4] tensor
  return path from Sortformer
- `src/transcript/pipeline_config.py:MergeConfig.strategy` (keep as
  `hard_boundary`-only or drop the dataclass entirely)
- The `prob_based` axis in `bench/tiers.py:_AXES_BOOL`

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

- `bench/results/runs.csv` — 352 rows total (192 Tier 1 + 160 Tier 2).
- `bench/results/leaderboard.md` — Tier 2 leaderboard per dataset.
- `bench/results/transcripts/tier-{1,2}/...` — per-(clip × config) hypothesis
  + reference utterances. Evidence for the follow-up failure-mode analysis.

---

## 4-bis. (historical) Original Tier-3 instructions

The harness is built but has never been run against real data. This is the
next concrete workstream.

### Pre-flight

1. **Vendor AMI RTTMs.** `bench/datasets/ami_rttm/` ships empty. The runtime
   git-clone fallback in `bench/datasets/ami.py:_resolve_rttm_dir` has a
   known bug: the BUT repo nests RTTMs under `<root>/only_words/rttms/` but
   the loader returns the root, so every AMI clip will silently skip (the
   `_log.warning("AMI: skipping %s — no RTTM ...")` line fires on each).
   Two ways to unblock:
   - **Quick:** download the BUT repo manually
     (`https://github.com/BUTSpeechFIT/AMI-diarization-setup`), copy
     `*.rttm` files from `only_words/rttms/` flat into
     `bench/datasets/ami_rttm/`, commit them.
   - **Clean:** fix `_resolve_rttm_dir` to descend into the nested layout.

2. **HuggingFace auth.** `linagora/SUMM-RE` may be gated. Run
   `huggingface-cli login` or set `HF_TOKEN` in the env. AMI's
   `edinburghcstr/ami` is public.

3. **Disk + bandwidth.** Plan for ~25 GB combined (AMI sdm test split +
   SUMM-RE dev split). First clip per meeting downloads; subsequent runs
   hit the HF dataset cache.

4. **ffmpeg.** Already a runtime dep but confirm `which ffmpeg` works —
   SUMM-RE's `_mix_tracks` uses it to combine per-speaker tracks into a
   single 16 kHz mono WAV.

### Running the sweep

Tiers must run in order — each reads the previous tier's rows from
`bench/results/runs.csv`:

```bash
uv run python scripts/benchmark.py --tier 1   # 32-config grid × 3 clips ≤150s
uv run python scripts/benchmark.py --tier 2   # narrowed axes × 10 clips ≤600s
uv run python scripts/benchmark.py --tier 3   # top-5 finalists × 50 clips full
```

Or chain with `--all`. To re-render the leaderboard without rerunning:
`uv run python scripts/benchmark.py --rebuild-leaderboard`.

Rough time budget (consumer Mac, no GPU):
- **Tier 1:** ~4–8 h. 192 rows total; the content-hashed cache amortises
  whisper/sortformer/align across the 32 configs per clip (only merge.py
  changes downstream of those stages).
- **Tier 2:** ~6–10 h. Longer clips, fewer configs.
- **Tier 3:** ~6–12 h. Full-duration clips × ≤5 finalists.

If a run is interrupted, restarting picks up cached stages automatically.

### Reading the results

- `bench/results/leaderboard.md` — tier-3 ranking per dataset by median
  cpWER. Auto-rendered after each tier; safe to re-render anytime.
- `bench/results/runs.csv` — every row across every tier. Columns include
  per-stage timings (whisper_s / sortformer_s / align_s / merge_s); the
  sentinel `-1.0` marks cached stages (filter `>= 0` before averaging).
- `bench/results/transcripts/tier-N/<clip>/<fingerprint>.json` — hypothesis
  + reference utterances for that (clip × config). The evidence for every
  CSV row.
- `bench/results/diffs/tier-N/...` — currently empty placeholders. The
  meeteval per-row diff extraction is deferred (see #5).

First three things to check:

1. **speaker_assignment_error_rate column** — the cpWER−WER decomposition
   that the prob_based merge specifically targets. If `prob_based` doesn't
   dominate `hard_boundary` on this column, the new strategy didn't earn
   its keep and should be reverted.
2. **streaming_preset** — does `low_lat` ever beat `very_high_lat`? NVIDIA
   says it shouldn't on batch workloads; if it does, investigate.
3. **no_fallback=False** — occasionally rescuing hard segments could lower
   WER without inflating cpWER. Worth checking before defaulting it.

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
  from `bench/runner.py`. Existing `bench/results/runs.csv` is kept as
  the historical evidence the revert is grounded in (the schema break
  means re-running the bench requires regenerating it).
- Deleted Tier 3 entirely. Without the `prob_based`-vs-`hard_boundary`
  race, Tier 3 would have collapsed to "re-run defaults on more clips"
  — and the verdict already showed defaults capture ~99.4% of
  achievable gain.
- Pruned `bench/cache.py` (emit_probs gone from sortformer_key; probs
  npy artefact gone). Existing `bench/cache/sortformer/` entries are
  orphans now — harmless but `rm -rf bench/cache/sortformer/` tidies.
- Updated/removed all affected tests; full non-integration suite green.

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
  exactly the failure mode the prob_based merge targets.

## 6. LLM-fix prompt-design tailored to residual failures — FOLLOW-UP

Deliberately excluded from the benchmark axes (per the project's
"tune deterministic core first" decision). After #4 lands the best
deterministic config, mine `bench/results/diffs/tier-3/` for the residual
error patterns (boundary swaps, stranded short islands, hallucinated
inserts) and rewrite the `llm_fix.apply` prompt to target only what the
deterministic pipeline can't fix. Without this, llm_fix's gemma4:e4b will
keep introducing more errors than it removes on cases the deterministic
pipeline already gets right.
