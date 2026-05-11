import json
import urllib.error

from transcript import llm_fix
from transcript.models import Word


def _wpairs():
    return [
        (Word(" Salut", 0.0, 0.4), "Speaker 1"),
        (Word(" Chouchou", 0.4, 0.9), "Speaker 1"),
        (Word(" ?", 0.9, 1.0), "Speaker 2"),  # boundary slip — index 2 should be Speaker 1
        (Word(" Car", 1.0, 1.3), "Speaker 2"),
    ]


def _mock_ollama(mocker, llm_response_text: str):
    """Patch urlopen to return an Ollama-shaped envelope: {"response": <text>}."""
    mock = mocker.patch("transcript.llm_fix.urllib.request.urlopen")
    mock.return_value.__enter__.return_value.read.return_value = (
        json.dumps({"response": llm_response_text}).encode()
    )
    return mock


def test_is_available_false_when_ollama_missing(mocker):
    mocker.patch("transcript.llm_fix.shutil.which", return_value=None)
    assert llm_fix.is_available() is False


def test_is_available_true_when_ollama_present(mocker):
    mocker.patch("transcript.llm_fix.shutil.which", return_value="/opt/homebrew/bin/ollama")
    assert llm_fix.is_available() is True


def test_apply_applies_flips(mocker):
    pairs = _wpairs()
    # The model decides word 2 (the "?") was misattributed and should be Speaker 1
    _mock_ollama(mocker, json.dumps({"flips": [{"i": 2, "spk": 1}]}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert [spk for _, spk in out] == ["Speaker 1", "Speaker 1", "Speaker 1", "Speaker 2"]


def test_apply_no_flips_means_unchanged(mocker):
    pairs = _wpairs()
    _mock_ollama(mocker, json.dumps({"flips": []}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_strips_markdown_fences(mocker):
    pairs = _wpairs()
    fenced = "```json\n" + json.dumps({"flips": [{"i": 2, "spk": 1}]}) + "\n```"
    _mock_ollama(mocker, fenced)
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out[2][1] == "Speaker 1"


def test_apply_ignores_out_of_range_indices(mocker):
    """The model occasionally hallucinates indices past the input length; drop them silently."""
    pairs = _wpairs()
    _mock_ollama(mocker, json.dumps({"flips": [{"i": 99, "spk": 1}, {"i": 2, "spk": 1}]}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    # Out-of-range dropped, in-range applied
    assert out[2][1] == "Speaker 1"
    assert len(out) == len(pairs)


def test_apply_falls_through_on_http_error(mocker):
    pairs = _wpairs()
    mocker.patch(
        "transcript.llm_fix.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "http://localhost:11434/api/generate", 500, "Server Error", {}, None
        ),
    )
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_falls_through_when_daemon_not_running(mocker):
    """Connection refused → URLError → graceful fallthrough."""
    pairs = _wpairs()
    mocker.patch(
        "transcript.llm_fix.urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    )
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_falls_through_on_timeout(mocker):
    pairs = _wpairs()
    mocker.patch(
        "transcript.llm_fix.urllib.request.urlopen",
        side_effect=TimeoutError("timed out"),
    )
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_falls_through_on_invalid_json(mocker):
    pairs = _wpairs()
    _mock_ollama(mocker, "not json at all")
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_falls_through_on_missing_flips_key(mocker):
    """If the model emits a JSON object without the `flips` key, treat as parse failure."""
    pairs = _wpairs()
    _mock_ollama(mocker, json.dumps({"other": "shape"}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_falls_through_on_flips_wrong_type(mocker):
    """Regression for the original bug: model emits a single object instead of {flips: [...]}"""
    pairs = _wpairs()
    _mock_ollama(mocker, json.dumps({"i": 0, "t": 0.12, "txt": " Salut", "spk": 1}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out == pairs


def test_apply_handles_empty_input():
    assert llm_fix.apply([], language="fr", num_speakers=2) == []


def test_apply_round_trips_unknown_speaker(mocker):
    pairs = [(Word(" hi", 0.0, 0.2), "Speaker 1")]
    _mock_ollama(mocker, json.dumps({"flips": [{"i": 0, "spk": 0}]}))
    out = llm_fix.apply(pairs, language="fr", num_speakers=2)
    assert out[0][1] == "Unknown"


def test_apply_sends_schema_constrained_request(mocker):
    """Body must include the JSON Schema in `format`, not the bare string 'json'.
    Without the schema, Gemma 4 E4B returns a single dict instead of the {flips: [...]} envelope."""
    pairs = _wpairs()
    mock = _mock_ollama(mocker, json.dumps({"flips": []}))
    llm_fix.apply(pairs, language="fr", num_speakers=2)
    sent_request = mock.call_args.args[0]
    body = json.loads(sent_request.data.decode())
    assert body["model"] == "gemma4:e4b"
    assert isinstance(body["format"], dict), "format must be a JSON Schema object, not a string"
    assert body["format"]["type"] == "object"
    assert "flips" in body["format"]["required"]
    assert "Language: fr" in body["prompt"]
