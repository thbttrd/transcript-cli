import numpy as np

from bench.cache import (
    align_key,
    audio_sha1,
    load_align,
    load_sortformer,
    load_whisper,
    save_align,
    save_sortformer,
    save_whisper,
    whisper_key,
)
from transcript.models import Turn, Word
from transcript.pipeline_config import DiarizeConfig, TranscribeConfig


def _make_wav(tmp_path, content=b"\0\0\0\0"):
    wav = tmp_path / "in.wav"
    wav.write_bytes(content)
    return wav


def test_audio_sha1_is_deterministic_and_hex(tmp_path):
    wav = _make_wav(tmp_path)
    h1 = audio_sha1(wav)
    h2 = audio_sha1(wav)
    assert h1 == h2
    assert len(h1) == 40
    assert all(c in "0123456789abcdef" for c in h1)


def test_whisper_key_changes_only_when_relevant_fields_change(tmp_path):
    wav = _make_wav(tmp_path)
    cfg_a = TranscribeConfig(model="large-v3", language="fr")
    cfg_b = TranscribeConfig(model="large-v3", language="fr", no_fallback=False)
    cfg_c = TranscribeConfig(model="large-v3", language="fr")
    assert whisper_key(wav, cfg_a) != whisper_key(wav, cfg_b)
    assert whisper_key(wav, cfg_a) == whisper_key(wav, cfg_c)


def test_save_and_load_whisper_roundtrips(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = TranscribeConfig(language="fr")
    words = [Word(" hi", 0.0, 0.5), Word(" there", 0.5, 1.0)]
    save_whisper(wav, cfg, words, cache_dir=tmp_path)
    assert load_whisper(wav, cfg, cache_dir=tmp_path) == words


def test_load_whisper_returns_none_on_miss(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = TranscribeConfig(language="fr")
    assert load_whisper(wav, cfg, cache_dir=tmp_path) is None


def test_save_and_load_sortformer_roundtrips_without_probs(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = DiarizeConfig()
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    save_sortformer(wav, cfg, turns, probs=None, cache_dir=tmp_path)
    loaded_turns, loaded_probs = load_sortformer(wav, cfg, cache_dir=tmp_path)
    assert loaded_turns == turns
    assert loaded_probs is None


def test_save_and_load_sortformer_roundtrips_with_probs(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = DiarizeConfig(emit_probs=True)
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    probs = np.random.RandomState(0).rand(20, 4).astype(np.float32)
    save_sortformer(wav, cfg, turns, probs=probs, cache_dir=tmp_path)
    loaded_turns, loaded_probs = load_sortformer(wav, cfg, cache_dir=tmp_path)
    assert loaded_turns == turns
    assert loaded_probs is not None
    np.testing.assert_array_equal(loaded_probs, probs)


def test_align_key_includes_whisper_hash(tmp_path):
    wav = _make_wav(tmp_path)
    cfg_a = TranscribeConfig(language="fr")
    cfg_b = TranscribeConfig(language="fr", no_fallback=False)
    h_a = align_key(wav, whisper_key(wav, cfg_a), language="fr")
    h_b = align_key(wav, whisper_key(wav, cfg_b), language="fr")
    assert h_a != h_b


def test_save_and_load_align_roundtrips(tmp_path):
    wav = _make_wav(tmp_path)
    words = [Word(" hi", 0.0, 0.5)]
    save_align(wav, "wkey", "fr", words, cache_dir=tmp_path)
    loaded = load_align(wav, "wkey", "fr", cache_dir=tmp_path)
    assert loaded == words
