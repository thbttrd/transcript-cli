from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from transcript import align, audio, diarize, llm_fix, merge, transcribe
from transcript.models import Meta, Turn, Utterance
from transcript.pipeline_config import PipelineConfig
from transcript.progress import Progress


def run(
    *,
    audio_path: Path,
    config: PipelineConfig,
    with_diarization: bool = True,
    progress: Progress | None = None,
) -> tuple[list[Utterance], Meta]:
    progress = progress or Progress(quiet=True)

    progress.step("preparing audio")
    wav, duration = audio.prepare(audio_path)
    progress.done("preparing audio")

    is_temp_wav = wav != audio_path
    try:
        diarize_cfg = config.diarize
        if config.merge.strategy == "prob_based":
            diarize_cfg = replace(diarize_cfg, emit_probs=True)

        if with_diarization:
            progress.step("transcribing + diarizing (parallel)")
            with ThreadPoolExecutor(max_workers=2) as ex:
                tx_fut = ex.submit(transcribe.run, wav, config=config.transcribe)
                diar_fut = ex.submit(diarize.run, wav, config=diarize_cfg)
                words, detected_lang = tx_fut.result()
                turns, probs = diar_fut.result()
            progress.done("transcribing + diarizing (parallel)")
        else:
            progress.step("transcribing")
            words, detected_lang = transcribe.run(wav, config=config.transcribe)
            turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
            probs = None
            progress.done("transcribing")

        if config.align.enabled and align.is_available() and words:
            progress.step("aligning words")
            words = align.run(wav, words, language=detected_lang)
            progress.done("aligning words")

        word_speakers = merge.assign_speakers(
            words, turns, strategy=config.merge.strategy, probs=probs
        )
        if with_diarization and config.llm_fix.enabled and llm_fix.is_available():
            progress.step("LLM cleanup")
            word_speakers = llm_fix.apply(
                word_speakers,
                language=detected_lang,
                num_speakers=config.diarize.num_speakers,
            )
            progress.done("LLM cleanup")

        progress.step("merging")
        utterances = merge.collapse(word_speakers)
        progress.done("merging")

        speaker_count = len({t.speaker for t in turns}) if turns else 0
        meta = Meta(
            filename=audio_path.name,
            duration=duration,
            model=config.transcribe.model,
            language=detected_lang,
            speaker_count=speaker_count,
            diarizer=diarize.DIARIZER_LABEL if with_diarization else None,
        )

        return utterances, meta
    finally:
        if is_temp_wav:
            wav.unlink(missing_ok=True)
