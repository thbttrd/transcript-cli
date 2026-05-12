# Performance Measurement Workflow — Design

- **Status:** Draft (awaiting user review)
- **Date:** 2026-05-11
- **Project directory:** `/Users/thibauttroude/Codes/sandbox/transcript-cli/`
- **Predecessor design:** `2026-05-09-transcript-cli-design.md`

## Summary

A reproducible, on-disk-cached hyperparameter-search harness for `transcript-cli` that runs the deterministic transcription pipeline against two real-world datasets (AMI for credibility, SUMM-RE for relevance to the French voice-memo use case), measures **cpWER** (concatenated minimum-permutation WER) plus diagnostic decompositions, and identifies the best deterministic configuration through a three-tier search (cheap exploratory grid → narrowed grid → definitive numbers on the full subset).

Replaces the current "eyeball one voice-memo" iteration workflow that the codebase's `docs/todo.md` calls out as the main blocker for principled improvements to the pipeline.

## Goals

- Quantify pipeline quality with a single primary number per dataset (cpWER) plus diagnostic decompositions (WER, DER, speaker-assignment error).
- Make every meaningful hyperparameter of the deterministic pipeline a first-class config field, so the search space is enumerable and the harness is the only thing that needs to change to add a knob.
- Cache stage outputs aggressively so re-running after a code change in `merge.py` (or any downstream stage) takes minutes, not hours.
- Run the harness from a single command (`python scripts/benchmark.py --tier 1`), with reproducible outputs (git-sha-pinned CSV rows).
- Surface results both as a flat CSV (for pandas / spreadsheet drill-down) and an auto-generated leaderboard markdown (for human reading).
- Implement TODO #1 (probability-based per-word speaker assignment) as part of this work, so the search can compare hard-boundary vs prob-based merge head-to-head.

## Non-goals

- Real-time / streaming evaluation (the CLI is batch-only).
- Confidence intervals / bootstrap analysis (deferred — useful but not load-bearing for picking among ~5 finalists).
- Multi-objective Pareto search across quality and runtime (single objective: cpWER; runtime is recorded but not optimised).
- Bayesian / Optuna-style smart search (the parameter space is small enough — 32 configs — that grid + tiering is enough).
- Tuning `llm_fix` (intentionally excluded — deterministic pipeline gets locked first; llm_fix revisited as a separate workstream only if the deterministic pipeline still has residual errors).
- Sweeping the whisper model size (frozen at `large-v3` — errors propagate downstream, no operating point prefers a smaller model on the joint metric).

## Decisions log (from brainstorming)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Primary purpose | Both AMI (credibility) + SUMM-RE (relevance) | Two-dataset harness — AMI for comparable numbers vs published baselines, SUMM-RE for the real French voice-memo use case |
| 2 | Knob scope | Maximal — including prob-based merge (implements TODO #1) | Answers the actual unresolved question in todo.md (does prob-based assignment beat hard boundary?) |
| 3 | Search strategy | Tiered: full grid on short clips → narrowed grid on medium clips → finalists on full subset | Cost-efficient way to identify which knobs move the needle |
| 4 | Delivery | `scripts/benchmark.py` + `bench/` module + new `bench` extras group | Keeps shipped CLI lean; bench deps installed only by developers |
| 5 | Caching | On-disk content-hashed cache (JSON + NumPy `.npy`, no pickle) | Re-running the same config or downstream-only edits is ~instant; pickle avoided for safety |
| 6 | LLM cleanup | Excluded from search axes | Tune deterministic core first |
| 7 | Whisper model | Frozen at `large-v3` | Upstream-stage error propagation — no joint-metric operating point prefers a smaller model |
| 8 | Whisper temperature | Dropped as axis | Adds noise without signal in the deterministic regime; `no_fallback` already captures the meaningful choice |
| 9 | Primary metric | cpWER via meeteval | Standard joint metric for ASR+diarization; charges substitution for "right word, wrong speaker" |
| 10 | AMI mic config | `sdm` (single distant mic) | Closer acoustic match to voice-memo conditions than headset-mic `ihm` |
| 11 | SUMM-RE split | `dev` only for v1 | Manually transcribed; `test` reserved so we never train/select on it |

## Pipeline graph + parameter map

```
                              audio file
                                   │
                          audio.prepare()
                                   │  (16 kHz mono WAV — frozen)
                                   ▼
                          ┌────────┴────────┐
                          ▼                 ▼
                  transcribe.run(wav)   diarize.run(wav)
                  ┌─────────────────┐   ┌──────────────────────┐
                  │ TUNABLE:        │   │ TUNABLE:             │
                  │  • no_fallback  │   │  • streaming_preset  │
                  │  • suppress_nst │   │ DERIVED:             │
                  │ FROZEN:         │   │  • num_speakers      │
                  │  • model        │   │    (from dataset)    │
                  │  • language     │   │  • emit_probs        │
                  │  • temperature  │   │    (auto if needed)  │
                  └─────────────────┘   └──────────────────────┘
                       │                            │
                  list[Word]                ┌───────┴───────┐
                       │                    ▼               ▼
                       │              list[Turn]    [T×4] probs
                       ▼                    │       (or None)
                  align.run() ◄─────────────┤
                  ┌──────────────┐          │
                  │ TUNABLE:     │          │
                  │  • enabled   │          │
                  └──────────────┘          │
                       │                    │
                  aligned Word[]            │
                       └─────────┬──────────┘
                                 ▼
                         merge.assign(...)
                     ┌────────────────────────┐
                     │ TUNABLE:               │
                     │  • strategy            │
                     │      hard_boundary     │
                     │      prob_based        │
                     │      (uses T×4 tensor) │
                     └────────────────────────┘
                                 │
                         word_speakers[]
                                 │
                                 ▼
                         llm_fix.apply(...)
                     ┌────────────────────────┐
                     │ FROZEN: enabled=False  │
                     │ (not part of search)   │
                     └────────────────────────┘
                                 │
                                 ▼
                         list[Utterance]   ← evaluation target
```

**Search axes (deterministic pipeline only):**

| Stage | Parameter | Values | Cost to vary |
|---|---|---|---|
| `transcribe` | `no_fallback` | `true`, `false` | High (whisper re-runs) |
| `transcribe` | `suppress_nst` | `true`, `false` | High (whisper re-runs) |
| `diarize` | `streaming_preset` | `very_high_lat`, `low_lat` | Medium (sortformer re-runs) |
| `align` | `enabled` | `true`, `false` | Medium (mms-300m re-runs) |
| `merge` | `strategy` | `hard_boundary`, `prob_based` | Negligible (pure Python) |

**Full Cartesian = 2 × 2 × 2 × 2 × 2 = 32 configs.** All on `large-v3`.

**Frozen / dataset-derived (not search axes):**

- `whisper.model` = `large-v3`
- `whisper.language` = dataset language (`en` for AMI, `fr` for SUMM-RE)
- `whisper.temperature` = `0.0`
- `whisper --max-len 1 --split-on-word` — structurally required for word boundaries
- `diarize.num_speakers` = dataset-provided per clip
- `align` model = `MahmoudAshraf97/mms-300m-1130-forced-aligner` (only viable option)
- `llm_fix.enabled` = `false`
- `audio.prepare` target = 16 kHz mono PCM16

## Config schema — `PipelineConfig`

A single nested dataclass tree mirrors the pipeline stages, one config namespace per stage. Lives in `src/transcript/pipeline_config.py`.

```python
@dataclass(frozen=True)
class TranscribeConfig:
    model: str = "large-v3"
    language: str | None = None       # dataset-set for bench
    temperature: float = 0.0
    no_fallback: bool = True
    suppress_nst: bool = True

@dataclass(frozen=True)
class DiarizeConfig:
    streaming_preset: Literal["very_high_lat", "low_lat"] = "very_high_lat"
    num_speakers: int | None = None   # dataset-set for bench
    emit_probs: bool = False          # auto-set true when merge.strategy == "prob_based"

@dataclass(frozen=True)
class AlignConfig:
    enabled: bool = True

@dataclass(frozen=True)
class MergeConfig:
    strategy: Literal["hard_boundary", "prob_based"] = "hard_boundary"

@dataclass(frozen=True)
class LLMFixConfig:
    enabled: bool = False             # frozen False for bench runs

@dataclass(frozen=True)
class PipelineConfig:
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    diarize:    DiarizeConfig    = field(default_factory=DiarizeConfig)
    align:      AlignConfig      = field(default_factory=AlignConfig)
    merge:      MergeConfig      = field(default_factory=MergeConfig)
    llm_fix:    LLMFixConfig     = field(default_factory=LLMFixConfig)

    def fingerprint(self) -> str:
        """Stable 12-char sha1 prefix — used as part of the cache key and CSV index."""
        return hashlib.sha1(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct from a flat dict (used by the tier generator)."""
        ...
```

`pipeline.run()` signature changes from the current 10-kwarg form to:

```python
def run(*, audio_path: Path, config: PipelineConfig,
        with_diarization: bool = True,
        progress: Progress | None = None) -> tuple[list[Utterance], Meta]:
    ...
```

The bench harness calls `pipeline.run()` and uses the returned `list[Utterance]` directly. The CLI calls `pipeline.run()` then runs a formatter over the result. Format is decoupled from pipeline execution.

**Per-stage refactors needed:**

- `transcribe.py` — read flags from `TranscribeConfig` instead of hard-coding `--no-fallback`, `--suppress-nst`. Add a `--temperature` flag in the subprocess command (defaults to 0).
- `diarize.py` — branch on `streaming_preset` (the two NVIDIA-published presets). Plumb `emit_probs` to `model.diarize(include_tensor_outputs=...)` and return the tensor alongside turns.
- `align.py` — read `enabled` from config; no behaviour change otherwise.
- `merge.py` — implement `prob_based` strategy as a sibling to the existing hard-boundary one. Function signature accepts the optional `[T×4]` probability tensor (used in prob mode only).
- `cli.py` — builds a `PipelineConfig` from argparse args; no user-visible CLI surface change.

## Search runner + on-disk cache

### File layout

```
bench/
├── __init__.py
├── runner.py           # Tier driver: build configs → execute → record
├── cache.py            # Content-hashed cache for whisper/sortformer/align outputs
├── metrics.py          # cpWER (primary) + WER + DER, via meeteval
├── tiers.py            # Tier-1 / Tier-2 / Tier-3 config generators
├── datasets/
│   ├── __init__.py
│   ├── base.py         # BenchClip dataclass + Dataset protocol
│   ├── ami.py          # AMI loader + reference RTTM
│   ├── summ_re.py      # SUMM-RE loader + track-mixing + synth RTTM
│   └── ami_rttm/       # Vendored RTTMs from BUTSpeechFIT/AMI-diarization-setup
└── results/
    ├── runs.csv                      # Append-only, one row per (clip × config × tier)
    ├── leaderboard.md                # Auto-generated from runs.csv
    ├── transcripts/<tier>/<clip>/<fp>.json   # hypothesis + reference utterances (git-ignored)
    └── diffs/<tier>/<clip>/<fp>.json         # meeteval word-level diff + speaker swaps (git-ignored)

scripts/
└── benchmark.py        # Thin CLI entry that calls into bench/runner.py
```

`bench/cache/` is created at runtime; **git-ignored**. `bench/results/` is **committed**.

### Tier definitions

| Tier | Clips/dataset | Configs | Est. wall (on M3) |
|---|---|---|---|
| 1 | 3 short (~2 min each) | full grid (32) | ~30 min |
| 2 | 10 medium (~10 min each) | top-K axes narrowed (~15 configs) | ~3 hr |
| 3 | full subset (~50/dataset) | 3–5 finalists | ~6–10 hr |

**Tier transitions are pure functions of CSV rows:**

- **Tier-2 generator** reads tier-1 rows from `runs.csv`. For each axis: compute the effect size = max cpWER swing across that axis's values, holding the other axes at the tier-1 best config. Keep axes whose effect ≥ **0.5 absolute cpWER points**; drop (lock at tier-1-best value) the rest. The narrowed grid is the Cartesian product over the kept axes' values, with dropped axes pinned. Typical outcome: 2–4 axes kept → 4–16 configs.
- **Tier-3 generator** reads tier-2 rows. Computes median cpWER across clips per config. Picks every config whose median is within **1.0 absolute cpWER points** of the best, capped at 5 finalists. If fewer than 3 are within that threshold, pad up to 3 by relaxing to within 2.0 points.

### Cache layout

Each cached stage hashes only the config fields that affect its own output, plus a content hash of the input audio. **No pickle** — JSON for structured data, NumPy `.npy` for raw tensors:

```
whisper/<hash>.json          hash = sha1(audio_sha1, "whisper",
                                  transcribe.model, transcribe.language,
                                  transcribe.temperature,
                                  transcribe.no_fallback, transcribe.suppress_nst)
                             content = list[Word] as JSON

sortformer/<hash>/           hash = sha1(audio_sha1, "sortformer",
                                  diarize.streaming_preset, diarize.emit_probs)
    turns.json               list[Turn] as JSON
    probs.npy                [T×4] float32 tensor (only present when emit_probs=True)

align/<hash>.json            hash = sha1(audio_sha1, "align",
                                  whisper_hash, transcribe.language)
                             content = aligned list[Word] as JSON
```

`merge` and `llm_fix` aren't cached — cheap and depend on cached upstream outputs.

`run_cached(audio, config)` is the harness's wrapper: for each stage, compute the cache key, check the cache, run-and-store on miss, return on hit. The actual pipeline functions stay cache-unaware — the bench harness wraps them.

### CLI entry

```bash
python scripts/benchmark.py --tier 1                    # ~30 min — full 32-config grid, 3 short clips/dataset
python scripts/benchmark.py --tier 2                    # ~3 hr — narrows from tier-1 rows in runs.csv
python scripts/benchmark.py --tier 3                    # ~6–10 hr — finalists from tier-2 rows
python scripts/benchmark.py --all                       # tier-1 → 2 → 3 sequentially
python scripts/benchmark.py --rebuild-leaderboard       # no compute, just re-generate .md
```

Tier N reads tier N−1 rows from `runs.csv` automatically — fails fast with a clear message if upstream rows are missing. Optional `--datasets ami summ-re` flag to filter; default is both.

## Datasets

Both datasets implement a common protocol:

```python
@dataclass(frozen=True)
class BenchClip:
    clip_id: str                  # e.g. "AMI:EN2002a", "SUMM-RE:001a_PARL"
    audio_path: Path              # 16 kHz mono WAV on local disk (pre-prepared)
    language: str                 # ISO 639-1, dataset-fixed ("en" / "fr")
    num_speakers: int             # known from dataset
    duration_s: float
    reference_rttm: Path
    reference_stm: Path

class Dataset(Protocol):
    name: str                     # "AMI" or "SUMM-RE"
    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]: ...
```

### AMI loader (`bench/datasets/ami.py`)

1. Download AMI from `edinburghcstr/ami` (HF dataset), `sdm` config (single distant mic — closer acoustic match to voice-memo conditions than `ihm`).
2. Per meeting, write a single 16 kHz mono WAV to `bench/cache/audio/ami/<meeting_id>.wav` (idempotent).
3. Use vendored reference RTTMs from `bench/datasets/ami_rttm/` (committed to the repo, ~few MB plain text from BUTSpeechFIT/AMI-diarization-setup).
4. Build per-meeting reference STM from AMI's manual transcripts.

### SUMM-RE loader (`bench/datasets/summ_re.py`)

1. Stream the `dev` split from `linagora/SUMM-RE` (manually transcribed; smaller than `test`; `test` reserved for tier-3 publication).
2. Group rows by `meeting_id` — each group is 3–4 single-speaker tracks.
3. For each meeting:
   - Resample every track to 16 kHz mono (source varies: 22 / 32 / 44 / 48 kHz).
   - Mix tracks with `ffmpeg -filter_complex amix=inputs=N:duration=longest:normalize=0` into a single 16 kHz mono WAV → `bench/cache/audio/summ_re/<meeting_id>.wav`.
   - Synthesise reference RTTM by emitting one `SPEAKER` line per (speaker, segment.start, segment.end) from the per-track segment metadata.
   - Synthesise reference STM by concatenating each track's per-word timestamps per speaker, sorted by time. Strip SUMM-RE's special markers (`@`, `*`, `+`) before writing.
4. Cache everything under `bench/cache/audio/summ_re/`. Re-running the loader is idempotent.

### Skip rules

Drop clips that fail `audio.prepare` (too short / corrupt), drop clips whose reference RTTM is empty, drop clips with `num_speakers > 4` (Sortformer 4-speaker cap). All drops are logged but not fatal.

### Storage budget

- AMI sdm audio (dev+test): ~5 GB
- SUMM-RE mixed audio (dev only): ~10–15 GB
- Whisper / sortformer / align caches: ~2–5 GB depending on grid coverage

Plan for ~25 GB of working set under `bench/cache/`.

## Metrics + reporting

### Primary metric

**cpWER** (concatenated minimum-permutation WER), computed via [`meeteval`](https://github.com/fgnt/meeteval). For each clip:

```python
# bench/metrics.py
def score(hyp_utterances: list[Utterance], reference_stm: Path) -> ClipMetrics:
    hyp_stm = utterances_to_stm(hyp_utterances)
    ref_stm = STM.load(reference_stm)
    cpwer_result = cpwer(reference=ref_stm, hypothesis=hyp_stm)
    wer_result   = speaker_agnostic_wer(hyp_stm, ref_stm)
    der_result   = der_from_rttm(hyp_utterances, reference_stm)
    return ClipMetrics(
        cpwer = cpwer_result.error_rate,
        wer   = wer_result.error_rate,
        der   = der_result.error_rate,
        speaker_assignment_error_rate = cpwer_result.error_rate - wer_result.error_rate,
    )
```

The fourth number — **speaker-assignment error rate** (`cpWER − WER`) — isolates "right word, wrong speaker." It's the most useful diagnostic for the prob-based merge work specifically.

### Normalisation

Applied identically to hypothesis and reference, centralised in `bench/metrics.py`:

- Lowercase, strip punctuation, collapse whitespace, NFC unicode-normalise.
- For SUMM-RE: also strip `@` laughter, `*` noise, `+` pause markers; expand common French contractions consistently.

### Results storage

Flat CSV at `bench/results/runs.csv`, append-only — never overwrite:

```
tier, dataset, clip_id, config_id, config_fingerprint,
no_fallback, suppress_nst, streaming_preset, align, merge_strategy,
cpwer, wer, der, speaker_assignment_error_rate,
runtime_s, whisper_s, sortformer_s, align_s, merge_s,
git_sha, started_at, host,
hypothesis_path, diff_path
```

- One row per `(clip × config × tier)`.
- Config flag columns denormalised for spreadsheet-pivot convenience.
- `hypothesis_path` and `diff_path` point to the per-row artefacts on disk (see next section).

### Per-row artefacts for failure-mode analysis

The CSV gives you metric numbers; the artefacts give you the *evidence* behind them. Persisted on disk for every `(clip × config × tier)` so post-hoc failure analysis — e.g. surveying where the best config still mislabels speakers, to design a future `llm_fix` prompt that targets those patterns — doesn't need to re-run the pipeline.

```
bench/results/
├── runs.csv                                      # numerical index (committed)
├── leaderboard.md                                # human-readable summary (committed)
├── transcripts/<tier>/<clip_id>/<config_fingerprint>.json
└── diffs/<tier>/<clip_id>/<config_fingerprint>.json
```

**`transcripts/<...>.json` schema:**

```json
{
  "clip_id":           "SUMM-RE:001a_PARL",
  "config_fingerprint":"a3f8c1b29e04",
  "hypothesis": [
    {"speaker": "Speaker 1", "start": 0.50, "end": 3.20, "text": "bonjour, nous allons parler de"},
    {"speaker": "Speaker 2", "start": 3.30, "end": 5.80, "text": "oui, exactement"},
    ...
  ],
  "reference": [
    {"speaker": "Speaker A", "start": 0.50, "end": 3.20, "text": "bonjour nous allons parler de"},
    ...
  ]
}
```

**`diffs/<...>.json` schema** — the meeteval-aligned word-level diff plus the cpWER speaker permutation:

```json
{
  "clip_id":           "SUMM-RE:001a_PARL",
  "config_fingerprint":"a3f8c1b29e04",
  "speaker_permutation": {"Speaker 1": "Speaker A", "Speaker 2": "Speaker B"},
  "word_ops": [
    {"op": "equal",  "ref_word": "bonjour",   "hyp_word": "bonjour",   "ref_speaker": "A", "hyp_speaker": "A"},
    {"op": "sub",    "ref_word": "allons",    "hyp_word": "allon",     "ref_speaker": "A", "hyp_speaker": "A"},
    {"op": "speaker_swap", "ref_word": "oui", "hyp_word": "oui",       "ref_speaker": "B", "hyp_speaker": "A"},
    ...
  ],
  "totals": {"sub": 14, "ins": 2, "del": 5, "speaker_swap": 8}
}
```

The `speaker_swap` op is the load-bearing one for future llm_fix work: it tells you which words got the text right but the speaker wrong, and where in the conversation they sit. Aggregating these across many clips reveals the *patterns* — boundary words, short interjections, mid-sentence flips — that a tailored prompt can address.

### Git policy for artefacts

- `bench/results/runs.csv` and `leaderboard.md` are **committed** (small, useful index).
- `bench/results/transcripts/` and `bench/results/diffs/` are **git-ignored by default** — tier-1 and tier-2 produce a lot of these and they'd bloat the repo.
- After a tier-3 run, you can opt-in commit the tier-3 subset for the failure-mode analysis work: `git add -f bench/results/{transcripts,diffs}/tier-3/`. The leaderboard.md links to those paths so reviewers can navigate to the per-clip evidence behind any leaderboard row.

### Leaderboard

`bench/results/leaderboard.md` is auto-generated from `runs.csv` by `bench.runner.generate_leaderboard()`. Filters to `tier == 3` rows for the headline table, aggregates by **median** (robust to one bad clip skewing) across clips per (dataset × config):

```markdown
# Benchmark leaderboard — last run <date> @ <sha>

## AMI (tier 3, N=50 clips, median)

| Rank | Config                                          | cpWER | WER  | DER  | Speaker-err | Runtime |
|------|-------------------------------------------------|-------|------|------|-------------|---------|
| 1    | align=on, merge=prob, sortformer=very-hi-lat    | 12.4  |  9.8 |  4.2 | 2.6         | 1.8×rt  |
| 2    | align=on, merge=hard, sortformer=very-hi-lat    | 14.1  |  9.8 |  4.2 | 4.3         | 1.7×rt  |
...

## SUMM-RE (tier 3, N=42 clips, dev split, median)

(same shape, French audio)
```

The CSV is the source of truth; the markdown is regeneratable from it at any time (`scripts/benchmark.py --rebuild-leaderboard`).

## Testing strategy

### Unit tests (fast, no audio, no network)

| Test file | What's verified |
|---|---|
| `tests/test_pipeline_config.py` | Defaults correct; `fingerprint()` stable across runs and changes only when a field changes; `from_dict` round-trips |
| `tests/test_bench_cache.py` | Cache key content-deterministic; miss writes to disk; hit reads from disk; invalidates on relevant config-field changes but NOT on irrelevant ones |
| `tests/test_bench_metrics.py` | cpWER on a hand-built golden example matches meeteval's expected output; speaker-assignment-error decomposition adds up; normalisation idempotent on AMI- and SUMM-RE-shaped strings |
| `tests/test_bench_tiers.py` | Tier-1 generates the full 32-config grid; tier-2 narrows correctly given seeded tier-1 results; tier-3 picks 3–5 finalists from tier-2 results |
| `tests/test_summ_re_loader.py` | Track-mixing produces a 16 kHz mono WAV of expected length; synth RTTM lines match per-track segments; empty-track meetings skipped with logged warning |
| `tests/test_merge_prob.py` | New prob-based merge: synthetic `[T×4]` tensor with sharp spike at speaker B for word frames assigns that speaker even when hard-boundary fallback says otherwise |
| `tests/test_bench_artefacts.py` | Transcript + diff JSON files are written for every `(clip × config × tier)` triple; their paths match the CSV row's `hypothesis_path` / `diff_path`; `speaker_swap` ops are emitted whenever hypothesis text matches reference text but their post-permutation speaker labels differ |

### Integration tests (gated by `pytest -m integration`)

- `tests/test_bench_smoke.py` — one tier-1 invocation on the existing `tests/fixtures/tiny.wav`, asserts a CSV row is appended and cache files appear. Validates the whole harness end-to-end without needing 25 GB of dataset audio.
- The existing `tests/test_pipeline_integration.py` is updated to use the new `PipelineConfig` API.

### Deliberately not tested

- meeteval correctness (third-party).
- HuggingFace `datasets` library correctness (third-party).
- Exact cpWER numbers on AMI/SUMM-RE (those are the *outputs* of the bench, not assertions about it).

## File layout — what changes

```
NEW FILES
─────────
src/transcript/pipeline_config.py
bench/__init__.py
bench/runner.py
bench/cache.py
bench/metrics.py
bench/tiers.py
bench/datasets/__init__.py
bench/datasets/base.py
bench/datasets/ami.py
bench/datasets/summ_re.py
bench/datasets/ami_rttm/...               (vendored from BUTSpeechFIT, ~MB)
scripts/benchmark.py
tests/test_pipeline_config.py
tests/test_bench_cache.py
tests/test_bench_metrics.py
tests/test_bench_tiers.py
tests/test_summ_re_loader.py
tests/test_merge_prob.py
tests/test_bench_smoke.py
docs/superpowers/specs/2026-05-11-perf-measurement-design.md   ← this spec

MODIFIED FILES
──────────────
pyproject.toml                            (new [project.optional-dependencies] bench)
src/transcript/pipeline.py                (accepts PipelineConfig; returns list[Utterance] + Meta)
src/transcript/transcribe.py              (reads flags from TranscribeConfig)
src/transcript/diarize.py                 (branches on streaming_preset; emit_probs)
src/transcript/align.py                   (reads enabled from AlignConfig)
src/transcript/merge.py                   (new prob_based strategy alongside existing hard_boundary)
src/transcript/cli.py                     (builds PipelineConfig from argparse, calls pipeline.run(config=...) then formats)
.gitignore                                (add bench/cache/, bench/results/transcripts/, bench/results/diffs/)

UNCHANGED
─────────
audio.py, llm_fix.py, formatters/*, doctor.py, progress.py, models.py
scripts/install.sh, scripts/dump_pipeline.py
```

### `pyproject.toml` change

```toml
[project.optional-dependencies]
bench = [
    "datasets==3.2.0",
    "meeteval==0.4.1",
    "pandas==2.2.3",
    "soundfile==0.12.1",
]
```

Installed only by developers: `uv sync --extra bench`. Shipped `transcript` CLI stays the same size.

## Operational notes

- **Reproducibility:** Whisper.cpp with `--no-fallback` and Sortformer streaming inference are both deterministic given the same audio and config. Two runs on the same machine should produce bit-identical CSV rows for the same `(clip_id, config_fingerprint)`. Different machines may differ slightly (CoreML/Metal nondeterminism); `host` column records which machine produced each row.
- **Crash recovery:** Each row is appended as soon as it's computed, so a crashed run loses only the in-flight clip. Resuming is automatic — already-cached configs short-circuit; already-recorded `(tier, clip, config_fingerprint)` triples are skipped on re-run.
- **Git hygiene:** `bench/cache/`, `bench/results/transcripts/`, and `bench/results/diffs/` are git-ignored. `bench/results/runs.csv` and `bench/results/leaderboard.md` are committed periodically as snapshots (with a `git_sha` column linking each row to the code state that produced it). After a tier-3 run, the tier-3 subset of transcripts + diffs can be opt-in committed for failure-mode analysis (`git add -f bench/results/{transcripts,diffs}/tier-3/`).
- **Downstream use — informs future llm_fix design:** the persisted diffs are the foundation for a follow-up workstream that tailors `llm_fix` prompts to the residual failure patterns of the best deterministic config. That workstream is *not* part of this design's scope — but the artefacts it will need are produced as a side effect of the bench runs here.

## Open questions to validate during implementation

- Exact `meeteval.wer.cpwer` API signature for the installed version (`0.4.1` is pinned but the API may have moved). Confirm against the version's docs at implementation time; fall back to manual STM construction if needed.
- AMI `sdm` config availability on `edinburghcstr/ami` — confirm it's the right HF config name versus `mdm` (multiple distant mics, an averaged variant). If `sdm` isn't directly downloadable, fall back to the closest single-channel option and document the deviation.
- SUMM-RE per-meeting track count: the dataset card says 3–4 speakers per meeting but some entries may be 5+ (exceeding Sortformer's 4-speaker cap). Skip rule handles this gracefully but log how many clips get dropped to verify the working dataset size.
- BUTSpeechFIT/AMI-diarization-setup license — vendoring its RTTMs into our repo needs license check. If their license forbids vendoring, fall back to a runtime-clone-on-first-use into `bench/cache/ami_rttm/`.

---

*This design is the input to the writing-plans skill, which will produce a step-by-step implementation plan.*
