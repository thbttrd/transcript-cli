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
