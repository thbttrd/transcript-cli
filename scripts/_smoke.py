"""One-off smoke driver: builds ONE ihm-mixed AMI WAV for manual listening.

Triggers the AMI ihm HF parquet download on first run (a few GB). Streams
the dataset filtering to one meeting; sums per-speaker headset audio at
each row's begin_time offset; peak-normalises; truncates to 300s. Reports
the output path + duration/energy stats so the user can manually verify
the mix sounds like a normal meeting recording before committing to Tier 1.
"""
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _emit(label: str) -> None:
    print(f"[smoke] {label}", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    import numpy as np
    import soundfile as sf
    from bench.datasets.ami import AMIDataset, _build_meeting_wav

    cache_dir = Path("bench/cache")
    ami = AMIDataset(cache_dir=cache_dir)
    rttm_files = sorted(p.stem for p in ami.rttm_dir.glob("*.rttm"))
    if not rttm_files:
        _emit("FAIL: no RTTMs in resolver-found dir")
        return 1

    # Pick the first meeting by sorted RTTM filename for reproducibility.
    meeting_id = rttm_files[0]
    out = cache_dir / "audio" / "ami" / f"{meeting_id}_300s.wav"
    _emit(f"target meeting: {meeting_id}")
    _emit(f"output path:    {out}")
    _emit("BEGIN ihm splice (300s cap)")

    t = time.time()
    try:
        _build_meeting_wav(meeting_id, out, max_duration_s=300.0)
    except Exception as e:
        _emit(f"FAIL {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    dt = time.time() - t
    _emit(f"DONE  ihm splice ({dt:.1f}s)")

    data, sr = sf.read(out)
    energy = abs(data).mean()
    peak = abs(data).max()
    nonzero_frac = (np.abs(data) > 1e-4).mean()
    _emit(f"WAV stats: dur={len(data)/sr:.1f}s  sr={sr}  "
          f"peak={peak:.3f}  mean_abs={energy:.4f}  "
          f"non-silent fraction={nonzero_frac:.1%}")
    _emit("ALL DONE — listen to the WAV at the path above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
