# TODO

Ideas to test, ordered by expected payoff.

## 1. Probability-based per-word speaker assignment

Replace the hard-boundary merge in `src/transcript/merge.py:_best_speaker` with a soft attribution computed from Sortformer's raw per-frame probabilities.

How:
- Pass `include_tensor_outputs=True` to `model.diarize(...)` in `src/transcript/diarize.py`. NeMo then returns a `[T_frames × 4]` probability tensor alongside the segments.
- In `merge.assign_speakers`, for each word, average the probabilities over the frames that fall within `[word.start, word.end]` (timestamps are now precise thanks to forced alignment) and take `argmax`.

Targets the two residual error patterns left after alignment:
- **Boundary timing off by 1–2 words** — fixed at the source: a word straddling a turn boundary gets attributed to whichever speaker the model was actually more confident about during the word's audio, not by midpoint geometry.
- **Stranded short islands** ("Bobby Lapointe" alone as S2) — suppressed: a 200 ms blip in the probability tensor won't stand alone if the surrounding frames don't truly peak for that speaker.

No new deps, no version bumps — Sortformer v1 already supports the tensor output mode. Probably obsoletes the LLM-cleanup path entirely on real audio.

## 2. Sortformer v2 / v2.1 (after a NeMo upgrade)

Attempted and reverted (see commit history). `nvidia/diar_streaming_sortformer_4spk-v2` needs `SortformerModules.spkcache_len` — a config field for the streaming speaker cache, introduced in a NeMo release newer than our pin `nemo_toolkit==2.2.1`.

Blocker: upgrading NeMo moves its own `transformers<=4.48.3` pin forward, which breaks the alignment loading path (`ctc-forced-aligner` calls `from_pretrained(dtype=...)` — works only on transformers ≥4.50, but our compat shim in `align.py:_load_model` uses `torch_dtype=` which is the pre-4.50 alias).

Path forward:
1. Pick a NeMo version that supports v2 *and* exposes a transformers pin ≤4.48.3 — unlikely; would need to read NeMo changelogs.
2. Or accept the transformers bump and update `align.py:_load_model` to use the new `dtype=` kwarg, rewriting the compat shim.
3. Or pin a specific git revision of `ctc-forced-aligner` from before they switched to `dtype=`.

Worth doing only if (1) above doesn't close the residual error gap.

## 3. Quantitative evaluation on real datasets

Replace the current "eyeball one voice-memo" workflow with reproducible Diarization Error Rate (DER) and Word Error Rate (WER) measurements against published transcripts.

Candidates:

- **`edinburghcstr/ami`** — ~29 GB total. Two configs: `ihm` (individual headset mics) and `sdm` (single distant mic), each ~half. 16 kHz parquet. Test split alone is ~12.6k short rows. Standard benchmark, lots of comparison numbers exist.
- **`linagora/SUMM-RE`** — ~93.8 GB total. Train is the bulk (~226 h of tracks). The manually-transcribed `dev` (~43 h) + `test` (~41 h) are what we actually want — together ~25–30 GB. Dataset viewer is disabled because parquet row groups exceed limits, so plan to stream rather than full-download.

Minimal harness:
- Script in `scripts/` that loads N random clips, runs the pipeline, compares to reference RTTM (for DER) and reference text (for WER).
- Print pre-fix / post-fix scores for each pipeline change (whisper params, post-proc YAML, alignment on/off, prob-based assignment on/off) so we stop arguing from one-shot voice-memo impressions.
