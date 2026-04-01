"""
benchmark.py — Variphi Crowd Analytics
Benchmarks YOLOv8 inference across ONNX Runtime and TensorRT backends.
Reports per-frame latency, FPS, and memory usage.

Usage:
        python benchmark.py --video data/test_video.mp4 --onnx weights/best.onnx --engine weights/best.engine
    python benchmark.py --video data/test_video.mp4 --onnx weights/best.onnx          # ONNX only
    python benchmark.py --video data/test_video.mp4 --engine weights/best.engine       # TensorRT only
"""

import argparse
import time
import csv
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import statistics

import cv2
import numpy as np

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging_utils import setup_logger

# Initialize logger
logger = setup_logger(name="benchmark", log_file="logs/benchmark_run.log")

# ── optional imports (fail gracefully if not installed) ──────────────────────
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not found. Install with: pip install ultralytics")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not found. Memory tracking disabled. pip install psutil")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── config ───────────────────────────────────────────────────────────────────

WARMUP_FRAMES   = 30    # frames to discard before recording (GPU warmup)
BENCHMARK_FRAMES = 300  # frames to actually measure
CONFIDENCE      = 0.25  # detection confidence threshold
IMG_SIZE        = 640   # model input resolution


# ── result containers ────────────────────────────────────────────────────────

@dataclass
class BackendResult:
    backend: str
    model_path: str
    total_frames: int = 0
    warmup_frames: int = WARMUP_FRAMES
    latencies_ms: list = field(default_factory=list)   # per-frame inference time

    # computed after run
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    fps: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    avg_detections: float = 0.0
    detection_counts: list = field(default_factory=list)

    def compute_stats(self):
        if not self.latencies_ms:
            return
        s = sorted(self.latencies_ms)
        n = len(s)
        self.mean_ms   = statistics.mean(s)
        self.median_ms = statistics.median(s)
        self.p95_ms    = s[int(0.95 * n)]
        self.p99_ms    = s[int(0.99 * n)]
        self.min_ms    = s[0]
        self.max_ms    = s[-1]
        self.std_ms    = statistics.stdev(s) if n > 1 else 0.0
        self.fps       = 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0
        if self.detection_counts:
            self.avg_detections = statistics.mean(self.detection_counts)

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Backend : {self.backend}",
            f"  Model   : {self.model_path}",
            f"  Frames  : {self.total_frames} measured  ({self.warmup_frames} warmup discarded)",
            f"{'─'*55}",
            f"  Mean latency  : {self.mean_ms:>8.2f} ms",
            f"  Median        : {self.median_ms:>8.2f} ms",
            f"  Std dev       : {self.std_ms:>8.2f} ms",
            f"  p95           : {self.p95_ms:>8.2f} ms",
            f"  p99           : {self.p99_ms:>8.2f} ms",
            f"  Min / Max     : {self.min_ms:>8.2f} ms / {self.max_ms:.2f} ms",
            f"{'─'*55}",
            f"  Throughput    : {self.fps:>8.2f} FPS",
            f"  Avg detections: {self.avg_detections:>8.2f} per frame",
        ]
        if self.gpu_memory_mb > 0:
            lines.append(f"  GPU memory    : {self.gpu_memory_mb:>8.1f} MB")
        if self.cpu_memory_mb > 0:
            lines.append(f"  CPU memory    : {self.cpu_memory_mb:>8.1f} MB")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ── benchmark runner ─────────────────────────────────────────────────────────

def get_gpu_memory_mb() -> float:
    """Returns currently allocated GPU memory in MB (CUDA only)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_cpu_memory_mb() -> float:
    """Returns current process RSS memory in MB."""
    if PSUTIL_AVAILABLE:
        import os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    return 0.0


def run_benchmark(
    model_path: str,
    backend: str,
    video_path: str,
    warmup: int = WARMUP_FRAMES,
    measure: int = BENCHMARK_FRAMES,
    device: int = 0,
) -> Optional[BackendResult]:
    """
    Runs inference on `measure` frames from the video and records latencies.
    `warmup` frames are run first and discarded to let the GPU settle.
    """
    if not ULTRALYTICS_AVAILABLE:
        logger.error(f"Cannot benchmark {backend} — ultralytics not installed.")
        return None

    if not Path(model_path).exists():
        logger.warning(f"{backend} model not found: {model_path}")
        return None

    result = BackendResult(backend=backend, model_path=model_path)

    logger.info(f"Loading {backend} model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load {backend} model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed = warmup + measure
    if total_video_frames > 0 and total_video_frames < needed:
        logger.warning(f"Video has only {total_video_frames} frames; need {needed}. Will loop.")

    frame_num  = 0
    phase      = "warmup"
    frames_run = 0  # frames run in current phase

    print(f"[INFO] Warming up ({warmup} frames)...", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            # loop video if too short
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # ── run inference, timed ──────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            preds = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                verbose=False,
                device=device,
            )
        except Exception as e:
            print(f"\n[ERROR] Inference failed on frame {frame_num}: {e}")
            break
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0

        if phase == "warmup":
            frames_run += 1
            if frames_run >= warmup:
                phase = "measure"
                frames_run = 0
                # snapshot memory after warmup (GPU caches are hot)
                result.gpu_memory_mb = get_gpu_memory_mb()
                result.cpu_memory_mb = get_cpu_memory_mb()
                print(" done.")
                print(f"[INFO] Measuring ({measure} frames)...", end="", flush=True)
        else:
            result.latencies_ms.append(elapsed_ms)
            n_dets = len(preds[0].boxes) if preds and preds[0].boxes is not None else 0
            result.detection_counts.append(n_dets)
            frames_run += 1
            result.total_frames += 1

            if frames_run % 50 == 0:
                print(f" {frames_run}", end="", flush=True)

            if frames_run >= measure:
                print(" done.")
                break

        frame_num += 1

    cap.release()
    result.compute_stats()
    return result


# ── comparison report ────────────────────────────────────────────────────────

def print_comparison(results: list[BackendResult]):
    """Prints a side-by-side speedup comparison if both backends ran."""
    valid = [r for r in results if r and r.fps > 0]
    if len(valid) < 2:
        return

    print("\n" + "=" * 55)
    print("  SPEEDUP COMPARISON")
    print("=" * 55)

    baseline = valid[0]  # ONNX (first)
    for r in valid[1:]:
        speedup_fps     = r.fps / baseline.fps if baseline.fps > 0 else 0
        latency_reduction = (1 - r.mean_ms / baseline.mean_ms) * 100 if baseline.mean_ms > 0 else 0
        print(f"  {r.backend} vs {baseline.backend}:")
        print(f"    FPS          : {baseline.fps:.1f} → {r.fps:.1f}  "
              f"({speedup_fps:.2f}× faster)")
        print(f"    Mean latency : {baseline.mean_ms:.2f} ms → {r.mean_ms:.2f} ms  "
              f"({latency_reduction:.1f}% reduction)")
    print("=" * 55)


# ── output writers ────────────────────────────────────────────────────────────

def save_csv(results: list[BackendResult], out_path: str):
    """Saves per-frame latency timeseries to CSV for plotting."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # find max frames across all results for column alignment
    max_frames = max((len(r.latencies_ms) for r in results if r), default=0)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # header row: one column per backend
        headers = ["frame"] + [r.backend for r in results if r]
        writer.writerow(headers)

        for i in range(max_frames):
            row = [i + 1]
            for r in results:
                if r and i < len(r.latencies_ms):
                    row.append(f"{r.latencies_ms[i]:.3f}")
                else:
                    row.append("")
            writer.writerow(row)

    print(f"[INFO] Per-frame latencies saved → {path}")


def save_summary_json(results: list[BackendResult], out_path: str):
    """Saves a summary JSON suitable for DESIGN.md or CI logging."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        if not r:
            continue
        d = asdict(r)
        del d["latencies_ms"]       # too large for summary
        del d["detection_counts"]
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Summary JSON saved       → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark YOLOv8 ONNX vs TensorRT for Variphi crowd analytics."
    )
    p.add_argument("--video",   required=True,               help="Path to test video file")
    p.add_argument("--onnx",    default=None,                help="Path to best.onnx")
    p.add_argument("--engine",  default=None,                help="Path to best.engine (TensorRT)")
    p.add_argument("--device",  type=int, default=0,         help="GPU device index (default 0)")
    p.add_argument("--warmup",  type=int, default=WARMUP_FRAMES,   help=f"Warmup frames (default {WARMUP_FRAMES})")
    p.add_argument("--frames",  type=int, default=BENCHMARK_FRAMES, help=f"Frames to measure (default {BENCHMARK_FRAMES})")
    p.add_argument("--out-dir", default="logs/benchmark",    help="Directory for output files")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.onnx and not args.engine:
        logger.error("Provide at least one of --onnx or --engine.")
        return

    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}")
        return

    print("\n" + "=" * 55)
    print("  Variphi Crowd Analytics — Inference Benchmark")
    print("=" * 55)
    print(f"  Video   : {args.video}")
    print(f"  Warmup  : {args.warmup} frames")
    print(f"  Measure : {args.frames} frames")
    print(f"  Device  : cuda:{args.device}")

    results = []

    # ── ONNX ────────────────────────────────────────────────────────────────
    if args.onnx:
        onnx_result = run_benchmark(
            model_path=args.onnx,
            backend="ONNX Runtime",
            video_path=args.video,
            warmup=args.warmup,
            measure=args.frames,
            device=args.device,
        )
        if onnx_result:
            print(onnx_result.summary())
            results.append(onnx_result)

    # ── TensorRT ─────────────────────────────────────────────────────────────
    if args.engine:
        trt_result = run_benchmark(
            model_path=args.engine,
            backend="TensorRT FP16",
            video_path=args.video,
            warmup=args.warmup,
            measure=args.frames,
            device=args.device,
        )
        if trt_result:
            print(trt_result.summary())
            results.append(trt_result)

    # ── comparison + outputs ─────────────────────────────────────────────────
    print_comparison(results)

    valid = [r for r in results if r]
    if valid:
        save_csv(valid, f"{args.out_dir}/latencies.csv")
        save_summary_json(valid, f"{args.out_dir}/summary.json")
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()