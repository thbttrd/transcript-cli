"""Output formatters. Each module exposes a render(utterances, meta, ...) -> str function.

Only md.render accepts a with_timestamps kwarg; the dispatcher's caller is responsible
for passing format-specific options conditionally (see pipeline.py for the pattern).
"""
from collections.abc import Callable

from transcript.formatters import json as _json
from transcript.formatters import md as _md
from transcript.formatters import srt as _srt
from transcript.formatters import txt as _txt

_REGISTRY: dict[str, Callable[..., str]] = {
    "md": _md.render,
    "json": _json.render,
    "srt": _srt.render,
    "txt": _txt.render,
}


def get(name: str) -> Callable[..., str]:
    if name not in _REGISTRY:
        raise ValueError(f"unknown format: {name!r} (expected one of {list(_REGISTRY)})")
    return _REGISTRY[name]
