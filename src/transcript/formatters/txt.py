from transcript.models import Meta, Utterance


def render(utterances: list[Utterance], meta: Meta) -> str:  # noqa: ARG001
    return "".join(f"{u.speaker}: {u.text}\n" for u in utterances)
