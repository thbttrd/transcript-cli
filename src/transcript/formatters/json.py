import json
from dataclasses import asdict

from transcript.models import Meta, Utterance


def render(utterances: list[Utterance], meta: Meta) -> str:
    payload = {
        "meta": asdict(meta),
        "utterances": [asdict(u) for u in utterances],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
