from transcript.models import Meta, Utterance


def _format_timestamp(seconds: float) -> str:
    """Format `seconds` as mm:ss for utterance timestamps."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _format_duration(seconds: float) -> str:
    """Format `seconds` as a compact human duration (12m34s, 1h23m45s)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def render(utterances: list[Utterance], meta: Meta, *, with_timestamps: bool = True) -> str:
    speaker_word = "speaker" if meta.speaker_count == 1 else "speakers"
    transcribed_with = f"whisper.cpp {meta.model} ({meta.language})"
    if meta.diarizer:
        transcribed_with += f" + {meta.diarizer}"
    lines: list[str] = [
        f"# {meta.filename}",
        "",
        (
            f"> Transcribed with {transcribed_with} · "
            f"{meta.speaker_count} {speaker_word} · "
            f"{_format_duration(meta.duration)}"
        ),
        "",
    ]
    for u in utterances:
        ts = f" [{_format_timestamp(u.start)}]" if with_timestamps else ""
        lines.append(f"## {u.speaker}{ts}")
        lines.append(u.text)
        lines.append("")
    return "\n".join(lines)
