from transcript.models import Meta, Utterance


def _srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render(utterances: list[Utterance], meta: Meta) -> str:  # noqa: ARG001 (meta unused but keeps signature uniform)
    blocks: list[str] = []
    for i, u in enumerate(utterances, start=1):
        blocks.append(
            f"{i}\n"
            f"{_srt_time(u.start)} --> {_srt_time(u.end)}\n"
            f"{u.speaker}: {u.text}\n"
        )
    return "\n".join(blocks)
