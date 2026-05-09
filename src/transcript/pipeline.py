from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transcript import audio, diarize, formatters, merge, transcribe
from transcript.models import Meta, Turn
from transcript.progress import Progress


def run(
    *,
    audio_path: Path,
    model: str,
    language: str | None,
    diarize: bool,
    num_speakers: int | None,
    format_name: str,
    with_timestamps: bool,
    progress: Progress | None = None,
) -> str:
    progress = progress or Progress(quiet=True)

    progress.step("preparing audio")
    wav, duration = audio.prepare(audio_path)
    progress.done("preparing audio")

    is_temp_wav = wav != audio_path
    try:
        if diarize:
            progress.step("transcribing + diarizing (parallel)")
            with ThreadPoolExecutor(max_workers=2) as ex:
                words_fut = ex.submit(transcribe.run, wav, model=model, language=language)
                turns_fut = ex.submit(diarize_module_run, wav, num_speakers)
                words = words_fut.result()
                turns = turns_fut.result()
            progress.done("transcribing + diarizing (parallel)")
        else:
            progress.step("transcribing")
            words = transcribe.run(wav, model=model, language=language)
            turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
            progress.done("transcribing")

        progress.step("merging")
        utterances = merge.assign(words, turns)
        progress.done("merging")

        speaker_count = len({t.speaker for t in turns}) if turns else 0
        meta = Meta(
            filename=audio_path.name,
            duration=duration,
            model=model,
            language=language or "auto",
            speaker_count=speaker_count,
        )

        render = formatters.get(format_name)
        # Only md supports with_timestamps; pass it conditionally
        if format_name == "md":
            return render(utterances, meta, with_timestamps=with_timestamps)
        return render(utterances, meta)
    finally:
        if is_temp_wav:
            wav.unlink(missing_ok=True)


# Indirection so tests can patch `transcript.pipeline.diarize.run`
def diarize_module_run(wav: Path, num_speakers: int | None):
    return diarize.run(wav, num_speakers=num_speakers)
