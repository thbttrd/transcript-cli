from transcript import doctor


def test_doctor_all_green(mocker, tmp_path):
    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "main")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "main").write_bytes(b"")
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)
    mocker.patch("transcript.doctor._nemo_importable", return_value=True)

    code, report = doctor.check()
    assert code == 0
    assert "✓" in report
    assert "✗" not in report


def test_doctor_reports_missing_binary(mocker, tmp_path):
    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "missing")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)
    mocker.patch("transcript.doctor._nemo_importable", return_value=True)

    code, report = doctor.check()
    assert code != 0
    assert "✗" in report
    assert "whisper" in report.lower()


def test_doctor_reports_missing_nemo(mocker, tmp_path):
    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "main")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "main").write_bytes(b"")
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)
    mocker.patch("transcript.doctor._nemo_importable", return_value=False)

    code, report = doctor.check()
    assert code != 0
    assert "nemo" in report.lower()
