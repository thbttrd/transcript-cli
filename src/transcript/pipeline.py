from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transcript import align, audio, diarize, diarize_diarizen, llm_fix, merge, transcribe
from transcript.models import Meta, Turn, Utterance
from transcript.pipeline_config import PipelineConfig
from transcript.progress import Progress

# Backend modules are looked up by DiarizeConfig.backend value. Each module must
# expose a `run(wav_path, *, config: DiarizeConfig) -> list[Turn]` callable plus
# a `DIARIZER_LABEL` string for the Meta record. Adding a new backend = adding
# a key here + a sibling module that satisfies that contract.
_DIARIZE_BACKENDS = {
    "sortformer": diarize,
    "diarizen": diarize_diarizen,
}


def _get_diarize_backend(name: str):
    if name not in _DIARIZE_BACKENDS:
        raise ValueError(
            f"unknown diarizer: {name!r} (expected one of {list(_DIARIZE_BACKENDS)})"
        )
    return _DIARIZE_BACKENDS[name]


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
    diar_module = _get_diarize_backend(config.diarize.backend)
    try:
        if with_diarization:
            progress.step("transcribing + diarizing (parallel)")
            with ThreadPoolExecutor(max_workers=2) as ex:
                tx_fut = ex.submit(transcribe.run, wav, config=config.transcribe)
                diar_fut = ex.submit(diar_module.run, wav, config=config.diarize)
                words, detected_lang = tx_fut.result()
                turns = diar_fut.result()
            progress.done("transcribing + diarizing (parallel)")
        else:
            progress.step("transcribing")
            words, detected_lang = transcribe.run(wav, config=config.transcribe)
            turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
            progress.done("transcribing")

        if config.align.enabled and align.is_available() and words:
            progress.step("aligning words")
            words = align.run(wav, words, language=detected_lang)
            progress.done("aligning words")

        word_speakers = merge.assign_speakers(words, turns)
        if with_diarization and config.merge.smooth_islands:
            word_speakers = merge.smooth_speaker_islands(
                word_speakers, max_island_words=config.merge.max_island_words
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
            diarizer=diar_module.DIARIZER_LABEL if with_diarization else None,
        )

        return utterances, meta
    finally:
        if is_temp_wav:
            wav.unlink(missing_ok=True)
