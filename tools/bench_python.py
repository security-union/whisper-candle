#!/usr/bin/env python3
"""PyTorch Whisper baseline benchmarks — the numbers the Rust port has to hit.

Usage:
    python tools/bench_python.py [--models tiny,base] [--devices cpu,mps]
                                 [--runs 3] [--out tests/fixtures/bench_python.json]

Methodology:
  - greedy decode (temperature=0, no beam), language pinned to "en"
  - fp32 on cpu, fp16 on mps (whisper's own per-device defaults)
  - clips: jfk.flac (11 s) and a ~60 s loop of it (built with ffmpeg on first run)
  - 1 warm-up + N timed runs, report best and median wall time
  - metric: RTF = audio_seconds / wall_seconds  (higher is better)
  - model load time reported separately, excluded from RTF
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audio_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def make_60s_clip(jfk: Path, dest: Path) -> Path:
    if not dest.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-stream_loop", "5", "-i", str(jfk),
             "-t", "60", "-ar", "16000", "-ac", "1", str(dest)], check=True)
    return dest


def bench_one(model, path: Path, duration: float, fp16: bool, runs: int):
    import whisper  # noqa: F401

    kwargs = dict(fp16=fp16, temperature=0.0, language="en", verbose=None)
    model.transcribe(str(path), **kwargs)  # warm-up
    times, text = [], ""
    for _ in range(runs):
        t0 = time.perf_counter()
        result = model.transcribe(str(path), **kwargs)
        times.append(time.perf_counter() - t0)
        text = result["text"]
    best, median = min(times), statistics.median(times)
    return {
        "wall_best_s": round(best, 3),
        "wall_median_s": round(median, 3),
        "rtf_best": round(duration / best, 2),
        "rtf_median": round(duration / median, 2),
        "text": text.strip()[:120],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-repo", default=str(HERE.parent.parent / "whisper"))
    ap.add_argument("--models", default="tiny,base")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=str(HERE.parent / "tests" / "fixtures" / "bench_python.json"))
    args = ap.parse_args()

    whisper_repo = Path(args.whisper_repo).resolve()
    sys.path.insert(0, str(whisper_repo))
    import torch
    import whisper

    jfk = whisper_repo / "tests" / "jfk.flac"
    clips = [("jfk_11s", jfk)]
    clip60 = make_60s_clip(jfk, HERE.parent / "tests" / "fixtures" / "jfk_60s.wav")
    clips.append(("jfk_60s", clip60))
    durations = {name: audio_duration(p) for name, p in clips}

    devices = []
    for d in args.devices.split(","):
        d = d.strip()
        if d == "mps" and not torch.backends.mps.is_available():
            print("mps not available, skipping")
            continue
        devices.append(d)

    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
            "runs": args.runs,
            "clip_durations_s": {k: round(v, 2) for k, v in durations.items()},
        },
        "results": [],
    }

    for device in devices:
        for model_name in [m.strip() for m in args.models.split(",")]:
            fp16 = device != "cpu"
            print(f"\n=== {model_name} on {device} (fp16={fp16}) ===")
            try:
                t0 = time.perf_counter()
                if device == "mps":
                    # sparse tensors can't be moved to MPS; densify alignment_heads
                    # (only used for word timestamps, which this bench doesn't exercise)
                    model = whisper.load_model(model_name, device="cpu")
                    model.register_buffer(
                        "alignment_heads", model.alignment_heads.to_dense(),
                        persistent=False)
                    model = model.to("mps")
                else:
                    model = whisper.load_model(model_name, device=device)
                load_s = time.perf_counter() - t0
                for clip_name, path in clips:
                    r = bench_one(model, path, durations[clip_name], fp16, args.runs)
                    entry = {"model": model_name, "device": device, "fp16": fp16,
                             "clip": clip_name, "load_s": round(load_s, 2), **r}
                    results["results"].append(entry)
                    print(f"  {clip_name}: best {r['wall_best_s']}s  "
                          f"RTF {r['rtf_best']}x  | {r['text'][:60]}...")
                del model
            except Exception as e:  # e.g. sparse-tensor issues on mps
                print(f"  FAILED: {type(e).__name__}: {e}")
                results["results"].append({"model": model_name, "device": device,
                                           "error": f"{type(e).__name__}: {e}"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    # markdown summary
    print("\n| model | device | clip | wall best (s) | RTF best |")
    print("|---|---|---|---|---|")
    for r in results["results"]:
        if "error" in r:
            print(f"| {r['model']} | {r['device']} | — | error | — |")
        else:
            print(f"| {r['model']} | {r['device']} | {r['clip']} | "
                  f"{r['wall_best_s']} | {r['rtf_best']}x |")


if __name__ == "__main__":
    main()
