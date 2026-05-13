"""Unit tests for scripts/benchmark.py — argv parsing + dispatch.

The script lives at scripts/benchmark.py and isn't a package, so we load
it via importlib for these tests rather than via a normal import.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("scripts_benchmark", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so mocker.patch("scripts_benchmark.X") can resolve
    # the symbol after the script body finishes its own imports.
    sys.modules["scripts_benchmark"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


def test_no_args_exits_with_error(script, capsys):
    with pytest.raises(SystemExit):
        script.main([])


def test_rebuild_leaderboard_calls_generator_and_skips_run(script, mocker, tmp_path):
    spy_gen = mocker.patch(
        "scripts_benchmark.runner.generate_leaderboard",
        return_value=tmp_path / "leaderboard.md",
    )
    spy_run = mocker.patch("scripts_benchmark.runner.run_one_tier")

    rc = script.main([
        "--rebuild-leaderboard",
        "--results-dir", str(tmp_path),
    ])

    assert rc == 0
    spy_gen.assert_called_once_with(results_dir=tmp_path)
    spy_run.assert_not_called()


def test_tier_1_dispatches_tier_1_configs_and_runs_once(script, mocker, tmp_path):
    spy_run = mocker.patch("scripts_benchmark.runner.run_one_tier")
    mocker.patch("scripts_benchmark.runner.generate_leaderboard",
                 return_value=tmp_path / "leaderboard.md")
    # Stop AMIDataset.__init__ from touching the network for the BUT RTTM repo.
    mocker.patch("scripts_benchmark.AMIDataset")
    mocker.patch("scripts_benchmark.SUMMREDataset")

    rc = script.main([
        "--tier", "1",
        "--cache-dir",   str(tmp_path / "cache"),
        "--results-dir", str(tmp_path / "results"),
    ])

    assert rc == 0
    assert spy_run.call_count == 1
    _, kwargs = spy_run.call_args
    assert kwargs["tier"] == 1
    assert kwargs["clip_count"] == 3
    assert kwargs["max_duration_s"] == 300.0
    # Tier 1 grid is 16 post-revert.
    assert len(kwargs["configs"]) == 16


def test_all_dispatches_tier_1_then_tier_2(script, mocker, tmp_path):
    spy_run = mocker.patch("scripts_benchmark.runner.run_one_tier")
    mocker.patch("scripts_benchmark.runner.generate_leaderboard",
                 return_value=tmp_path / "leaderboard.md")
    mocker.patch("scripts_benchmark.AMIDataset")
    mocker.patch("scripts_benchmark.SUMMREDataset")
    # Pre-populate runs.csv with a single tier-1 row so _configs_for_tier(2)
    # has something to feed tier_2_configs.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "runs.csv").write_text(
        "tier,dataset,clip_id,config_id,config_fingerprint,no_fallback,"
        "suppress_nst,streaming_preset,align,cpwer,wer,der,"
        "speaker_assignment_error_rate,runtime_s,whisper_s,sortformer_s,"
        "align_s,merge_s,git_sha,started_at,host,hypothesis_path,diff_path\n"
        "1,AMI,m1,abc,abc,True,True,very_high_lat,True,0.10,0.08,0.05,"
        "0.02,1.0,1.0,0.4,0.05,0.05,deadbee,0,host,h,d\n"
    )

    rc = script.main([
        "--all",
        "--cache-dir",   str(tmp_path / "cache"),
        "--results-dir", str(results_dir),
    ])

    assert rc == 0
    assert spy_run.call_count == 2
    assert spy_run.call_args_list[0].kwargs["tier"] == 1
    assert spy_run.call_args_list[1].kwargs["tier"] == 2


def test_tier_2_without_tier_1_rows_exits(script, mocker, tmp_path):
    mocker.patch("scripts_benchmark.runner.run_one_tier")
    mocker.patch("scripts_benchmark.AMIDataset")
    mocker.patch("scripts_benchmark.SUMMREDataset")

    with pytest.raises(SystemExit, match="tier 2 requires tier-1"):
        script.main([
            "--tier", "2",
            "--cache-dir",   str(tmp_path / "cache"),
            "--results-dir", str(tmp_path / "results"),
        ])


def test_datasets_flag_filters_constructed_datasets(script, mocker, tmp_path):
    mocker.patch("scripts_benchmark.runner.run_one_tier")
    mocker.patch("scripts_benchmark.runner.generate_leaderboard",
                 return_value=tmp_path / "leaderboard.md")
    ami_spy = mocker.patch("scripts_benchmark.AMIDataset")
    summ_re_spy = mocker.patch("scripts_benchmark.SUMMREDataset")

    script.main([
        "--tier", "1", "--datasets", "ami",
        "--cache-dir",   str(tmp_path / "cache"),
        "--results-dir", str(tmp_path / "results"),
    ])

    ami_spy.assert_called_once()
    summ_re_spy.assert_not_called()


def test_tier_3_is_rejected_by_argparse(script):
    with pytest.raises(SystemExit):
        script.main(["--tier", "3"])
