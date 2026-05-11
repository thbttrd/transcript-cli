from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transcript import align, audio, diarize, formatters, llm_fix, merge, transcribe
from transcript.models import Meta, Turn
from transcript.progress import Progress


def run(
    *,
    audio_path: Path,
    model: str,
    language: str | None,
    with_diarization: bool,
    num_speakers: int | None,
    format_name: str,
    with_timestamps: bool,
    with_align: bool = True,
    with_llm_fix: bool = False,
    progress: Progress | None = None,
) -> str:
    progress = progress or Progress(quiet=True)

    progress.step("preparing audio")
    wav, duration = audio.prepare(audio_path)
    progress.done("preparing audio")

    is_temp_wav = wav != audio_path
    try:
        if with_diarization:
            progress.step("transcribing + diarizing (parallel)")
            with ThreadPoolExecutor(max_workers=2) as ex:
                tx_fut = ex.submit(transcribe.run, wav, model=model, language=language)
                turns_fut = ex.submit(diarize_module_run, wav, num_speakers)
                words, detected_lang = tx_fut.result()
                turns = turns_fut.result()
            progress.done("transcribing + diarizing (parallel)")
        else:
            progress.step("transcribing")
            words, detected_lang = transcribe.run(wav, model=model, language=language)
            turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
            progress.done("transcribing")

        if with_align and align.is_available() and words:
            progress.step("aligning words")
            words = align.run(wav, words, language=detected_lang)
            progress.done("aligning words")

        word_speakers = merge.assign_speakers(words, turns)
        if with_diarization and with_llm_fix and llm_fix.is_available():
            progress.step("LLM cleanup")
            word_speakers = llm_fix.apply(
                word_speakers,
                language=detected_lang,
                num_speakers=num_speakers,
            )
            progress.done("LLM cleanup")

        progress.step("merging")
        utterances = merge.collapse(word_speakers)
        progress.done("merging")

        speaker_count = len({t.speaker for t in turns}) if turns else 0
        meta = Meta(
            filename=audio_path.name,
            duration=duration,
            model=model,
            language=detected_lang,
            speaker_count=speaker_count,
            diarizer=diarize.DIARIZER_LABEL if with_diarization else None,
        )

        render = formatters.get(format_name)
        if format_name == "md":
            return render(utterances, meta, with_timestamps=with_timestamps)
        return render(utterances, meta)
    finally:
        if is_temp_wav:
            wav.unlink(missing_ok=True)


def diarize_module_run(wav: Path, num_speakers: int | None):
    return diarize.run(wav, num_speakers=num_speakers)
