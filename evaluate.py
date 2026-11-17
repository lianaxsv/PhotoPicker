"""Reproducible image degradation experiment using a supplied reference image.

Usage: python evaluate.py image.jpg results.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from photo_quality.core import gamma_correct, adaptive_gamma

def metrics(reference: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    a = reference.astype(np.float64)
    b = output.astype(np.float64)
    mae = float(np.abs(a - b).mean())
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
    return mae, psnr

def evaluate(reference: np.ndarray) -> list[dict]:
    results = []
    for scale in [0.25, 0.4, 0.55, 0.7, 0.85]:
        dark = np.clip(reference.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        methods = {
            "uncorrected": dark,
            "fixed_gamma_0.8": gamma_correct(dark, 0.8),
            "fixed_gamma_0.6": gamma_correct(dark, 0.6),
            "adaptive_gamma": adaptive_gamma(dark)[0],
        }
        for name, image in methods.items():
            mae, psnr = metrics(reference, image)
            results.append({"darkening_scale":scale, "method":name, "mae":round(mae,4),
                            "psnr_db":round(psnr,4), "mean_brightness":round(float(image.mean()),4)})
    return results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("csv_output", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.reference))
    if image is None:
        raise SystemExit(f"Cannot read reference image: {args.reference}")
    rows = evaluate(image)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measured rows to {args.csv_output}")

if __name__ == "__main__":
    main()
