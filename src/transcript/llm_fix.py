"""LLM post-processing for diarization correction.

Calls a locally-served Ollama model (default: Gemma 4 E4B, an Apache-2.0 ~4.5 B
multilingual model that fits comfortably on 18 GB unified memory) over the HTTP
API at http://localhost:11434, sends the per-word speaker assignments as JSON,
and asks it to list ONLY the words whose speaker label is wrong. We then apply
the flips locally — far more reliable for small models than asking them to echo
the whole array back. Falls through on any failure but writes a diagnostic log
to `$TMPDIR/transcript-llm-fix.log` so silent failures can be inspected.
"""
import json
import re
import shutil
import sys
import urllib.error
import urllib.request

from transcript import _debug
from transcript.models import Word

_OLLAMA_BINARY = "ollama"
_OLLAMA_URL = "http://localhost:11434/api/generate"
# Edit if your local Ollama tag differs (e.g. `gemma4:26b-a4b`).
_MODEL = "gemma4:e4b"
_TIMEOUT_S = 60
_DEBUG_LOG = _debug.log_path("llm-fix")

# A "flips-only" contract: the model lists indices whose speaker label is wrong,
# instead of re-emitting the whole corrected array. Far easier for a 4.5 B model
# and the output is short enough that constrained-decoding via JSON Schema can
# pin its exact shape.
_SYSTEM_PROMPT = """\
You correct speaker-label errors in transcripts.

INPUT: a JSON array of words. Each word has:
  - i:   index (integer, starts at 0)
  - t:   timestamp in seconds (number)
  - txt: word text (string)
  - spk: current speaker label (integer; 1, 2, 3, 4, or 0 for Unknown)

YOUR JOB: find words whose spk is WRONG and list only those.

Heuristics for spotting wrong labels:
- A speaker rarely changes mid-sentence.
- Standalone punctuation (",", "?", ".") belongs to whoever said the words it terminates.
- A single word stranded between two same-speaker neighbors is almost always wrong.
- Questions and their answers are usually different speakers.
- Lists, songs, and recitations are usually one speaker.

OUTPUT FORMAT: a single JSON object exactly of this shape:
  {"flips": [{"i": <index>, "spk": <correct_speaker>}, ...]}

- Include ONLY words whose spk needs to change. Omit correct ones.
- Most labels are already correct — be conservative.
- If nothing needs flipping, output {"flips": []}.

Example: if input is [{"i":0,"t":0.0,"txt":" Hello","spk":1},{"i":1,"t":0.5,"txt":" hi","spk":2},{"i":2,"t":1.0,"txt":" there","spk":2}]
and "there" should actually be Speaker 1, output:
  {"flips":[{"i":2,"spk":1}]}"""

_USER_PROMPT = """\
Language: {language}
Number of speakers: {num_speakers}

List the wrong speaker labels in this array. Reply with ONLY the JSON object.

{payload}"""


def is_available() -> bool:
    return shutil.which(_OLLAMA_BINARY) is not None


def apply(
    word_speakers: list[tuple[Word, str]],
    *,
    language: str,
    num_speakers: int | None,
) -> list[tuple[Word, str]]:
    """Send word/speaker pairs to the local Ollama daemon. Returns input unchanged on any failure."""
    if not word_speakers:
        return word_speakers

    payload = [
        {"i": i, "t": round(w.start, 2), "txt": w.text, "spk": _spk_to_int(spk)}
        for i, (w, spk) in enumerate(word_speakers)
    ]
    user_prompt = _USER_PROMPT.format(
        language=language,
        num_speakers=num_speakers if num_speakers is not None else "unknown (treat as 2)",
        payload=json.dumps(payload, ensure_ascii=False),
    )

    body = json.dumps({
        "model": _MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": user_prompt,
        "format": _build_response_schema(len(word_speakers)),
        "stream": False,
        "options": {"temperature": 0},
        "keep_alive": "5m",
    }).encode()

    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            envelope_raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        _log_failure(f"HTTPError {e.code}", user_prompt, error_detail=e.read().decode(errors="replace"))
        return word_speakers
    except urllib.error.URLError as e:
        _log_failure(f"URLError ({e.reason}) — is `ollama serve` running?", user_prompt)
        return word_speakers
    except TimeoutError:
        _log_failure("TimeoutError", user_prompt)
        return word_speakers

    try:
        response_text = json.loads(envelope_raw).get("response", "")
    except json.JSONDecodeError:
        _log_failure("Ollama envelope was not JSON", user_prompt, response_text=envelope_raw)
        return word_speakers

    flips = _parse_flips(response_text, n_words=len(word_speakers))
    if flips is None:
        _log_failure("payload parse failure", user_prompt, response_text=response_text)
        return word_speakers

    corrected = list(word_speakers)
    for idx, new_spk_int in flips:
        w, _ = corrected[idx]
        corrected[idx] = (w, _int_to_spk(new_spk_int))

    _log_success(user_prompt, response_text, flips=len(flips), total=len(word_speakers))
    return corrected


def _log_failure(reason: str, prompt: str, *, response_text: str = "", error_detail: str = "") -> None:
    print(f"⚠ LLM cleanup failed ({reason}); falling back to un-corrected output. See {_DEBUG_LOG}.", file=sys.stderr)
    sections = [
        f"=== LLM FIX FAILURE: {reason} ===",
        f"model: {_MODEL}",
        f"endpoint: {_OLLAMA_URL}",
        "",
        "--- prompt ---",
        prompt,
    ]
    if response_text:
        sections += ["--- response ---", response_text]
    if error_detail:
        sections += ["--- error detail ---", error_detail]
    _debug.write(_DEBUG_LOG, "\n".join(sections))


def _log_success(prompt: str, response_text: str, *, flips: int, total: int) -> None:
    sections = [
        f"=== LLM FIX SUCCESS: {flips}/{total} words flipped ===",
        f"model: {_MODEL}",
        "",
        "--- prompt ---",
        prompt,
        "--- response ---",
        response_text,
    ]
    _debug.write(_DEBUG_LOG, "\n".join(sections))


def _build_response_schema(n_words: int) -> dict:
    """Ollama constrained-decoding schema. Without it, Gemma 4 E4B happily returns
    a single dict instead of the `{flips: [...]}` envelope we need."""
    return {
        "type": "object",
        "required": ["flips"],
        "additionalProperties": False,
        "properties": {
            "flips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["i", "spk"],
                    "additionalProperties": False,
                    "properties": {
                        "i": {"type": "integer", "minimum": 0, "maximum": n_words - 1},
                        "spk": {"type": "integer", "minimum": 0, "maximum": 4},
                    },
                },
            },
        },
    }


def _parse_flips(text: str, *, n_words: int) -> list[tuple[int, int]] | None:
    """Parse the LLM's `{"flips": [...]}` response. Returns list of (index, new_spk_int)
    or None on parse failure. Tolerates optional markdown fences."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("flips"), list):
        return None
    out: list[tuple[int, int]] = []
    for item in data["flips"]:
        if not isinstance(item, dict):
            return None
        idx = item.get("i")
        spk = item.get("spk")
        if not isinstance(idx, int) or not isinstance(spk, int):
            return None
        if 0 <= idx < n_words:
            out.append((idx, spk))
        # silently drop out-of-range — model occasionally hallucinates indices
    return out


def _spk_to_int(label: str) -> int:
    if label == "Unknown":
        return 0
    try:
        return int(label.split()[-1])
    except ValueError:
        return 0


def _int_to_spk(n: int) -> str:
    if not isinstance(n, int) or n <= 0:
        return "Unknown"
    return f"Speaker {n}"
