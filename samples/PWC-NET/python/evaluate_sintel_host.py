#!/usr/bin/env python3
"""Evaluate the algo.py speed path against MPI Sintel flow ground truth.

The workspace does not contain the CUDA-only ``nvof`` Python extension.  This
script therefore imports and calls the exact ``calc_of_speed`` function from
algo.py, while injecting a deterministic OpenCV Farneback implementation as
the ``nvof.calc`` compatibility layer.  The report labels this substitution
explicitly; it must not be mistaken for a measurement of CUDA nvof.calc.
"""
import argparse
import importlib.util
import math
import time
from pathlib import Path

import cv2
import numpy as np


class _TimeCache:
    def check_in(self):
        return True


class _Label:
    of_speed = 0.0


class _State:
    def __init__(self):
        self.of_last_img = None
        self.of_speed_list = []
        self.temp_of_speed = 0.0
        self.time_cache = _TimeCache()


class _FarnebackNvof:
    def __init__(self):
        self.last_flow = None
        self.last_flow_ms = 0.0

    def calc(self, previous, current, _unused):
        begin = time.perf_counter()
        self.last_flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            0.5, 3, 15, 3, 5, 1.2, 0)
        self.last_flow_ms = (time.perf_counter() - begin) * 1000.0
        return [self.last_flow]


def load_algo():
    path = Path(__file__).resolve().parents[1] / "src" / "algo.py"
    spec = importlib.util.spec_from_file_location("algo_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.cv2 = cv2
    module.np = np
    module.nvof = _FarnebackNvof()
    return module


def read_flo(path):
    with open(path, "rb") as f:
        if f.read(4) != b"PIEH":
            raise ValueError(f"not a Sintel .flo file: {path}")
        width = int(np.fromfile(f, dtype="<i4", count=1)[0])
        height = int(np.fromfile(f, dtype="<i4", count=1)[0])
        return np.fromfile(f, dtype="<f4", count=width * height * 2).reshape(height, width, 2)


def metrics(pred, truth, sums):
    valid = np.isfinite(truth).all(axis=2)
    delta = pred[valid].astype(np.float64) - truth[valid].astype(np.float64)
    epe = np.linalg.norm(delta, axis=1)
    dot = np.sum(pred[valid] * truth[valid], axis=1) + 1.0
    pn = np.sqrt(np.sum(pred[valid] ** 2, axis=1) + 1.0)
    tn = np.sqrt(np.sum(truth[valid] ** 2, axis=1) + 1.0)
    angle = np.arccos(np.clip(dot / (pn * tn), -1.0, 1.0)) * 180.0 / math.pi
    sums["pixels"] += int(epe.size)
    sums["epe"] += float(epe.sum())
    sums["angle"] += float(angle.sum())
    sums["bad3"] += int(np.count_nonzero(epe > 3.0))
    sums["bad5"] += int(np.count_nonzero(epe > 5.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.threads > 0:
        cv2.setNumThreads(args.threads)

    algo = load_algo()
    nvof = algo.nvof
    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for raw in f:
            if raw.strip() and not raw.lstrip().startswith("#"):
                fields = raw.split()
                if len(fields) >= 3:
                    rows.append((fields[0], fields[1], fields[2], fields[3] if len(fields) > 3 else ""))

    sums = {"pixels": 0, "epe": 0.0, "angle": 0.0, "bad3": 0, "bad5": 0}
    state = None
    label = _Label()
    last_scene = None
    flow_ms_sum = 0.0
    calc_ms_sum = 0.0
    count = 0
    overall_begin = time.perf_counter()
    for image0_path, image1_path, flow_path, scene in rows:
        if args.max_pairs and count >= args.max_pairs:
            break
        image0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
        image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        truth = read_flo(flow_path)
        if image0 is None or image1 is None:
            raise RuntimeError(f"cannot read {image0_path} or {image1_path}")
        if scene != last_scene:
            state = _State()
            # Match the first, unmeasured algo.py call on a scene's first frame.
            algo.calc_of_speed(state, label, image0)
            last_scene = scene
        begin = time.perf_counter()
        algo.calc_of_speed(state, label, image1)
        calc_ms = (time.perf_counter() - begin) * 1000.0
        flow_ms = nvof.last_flow_ms
        metrics(nvof.last_flow, truth, sums)
        flow_ms_sum += flow_ms
        calc_ms_sum += calc_ms
        count += 1
        if args.output:
            rows_out = f"{count},{scene},{flow_ms:.6f},{calc_ms:.6f},{label.of_speed:.6f}\n"
            rows.append  # keep the source manifest list alive; no-op for clarity
            with open(args.output, "a", encoding="utf-8") as out:
                out.write(rows_out)

    pixels = sums["pixels"]
    elapsed = (time.perf_counter() - overall_begin) * 1000.0
    print(f"host_backend=algo.py_with_farneback_compatibility_layer")
    print(f"opencv_threads={cv2.getNumThreads()}")
    print(f"pairs={count}")
    print(f"mean_epe={sums['epe'] / pixels if pixels else 0.0:.6f}")
    print(f"mean_angular_error_deg={sums['angle'] / pixels if pixels else 0.0:.6f}")
    print(f"bad3_percent={100.0 * sums['bad3'] / pixels if pixels else 0.0:.6f}")
    print(f"bad5_percent={100.0 * sums['bad5'] / pixels if pixels else 0.0:.6f}")
    print(f"flow_calc_avg_ms={flow_ms_sum / count if count else 0.0:.6f}")
    print(f"calc_of_speed_avg_ms={calc_ms_sum / count if count else 0.0:.6f}")
    print(f"overall_ms={elapsed:.3f}")


if __name__ == "__main__":
    main()
