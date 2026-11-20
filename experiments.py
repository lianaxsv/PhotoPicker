"""Reproducible controlled development experiments; no real-photo claims.

Run: python experiments.py --output results
Outputs: CSV files, sample PNGs, and a short summary.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

from photopicker import (Thresholds, adaptive_gamma, calibrate_threshold, clahe_enhance,
                         detect, gamma_correct, metrics, quality_score, rank_photos)


def reference_image(seed=0, size=256):
    """Artificial photo-like image with gradients, edges, colored objects, and texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    base = np.stack([
        65 + 0.35*x + 0.13*y,
        75 + 0.16*x + 0.26*y,
        95 + 0.12*x + 0.19*y,
    ], axis=2)
    base += rng.normal(0, 5, (size, size, 1))
    image = np.clip(base, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (34, 42), (155, 168), (30, 165, 240), -1)
    cv2.circle(image, (191, 172), 42, (215, 65, 40), -1)
    cv2.line(image, (0, 222), (255, 222), (255, 255, 255), 3)
    cv2.putText(image, "PHOTO", (33, 204), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (230, 225, 225), 2)
    return image


def darken(image, factor):
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def low_contrast(image, factor):
    return np.clip(127.5 + (image.astype(np.float32) - 127.5) * factor, 0, 255).astype(np.uint8)


def mae(reference, candidate):
    return float(np.mean(np.abs(reference.astype(np.float32) -
                                candidate.astype(np.float32))))


def psnr(reference, candidate):
    return float(cv2.PSNR(reference, candidate))


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    # Iteration 2: tune blur cutoff on synthetic examples, then evaluate held-out seeds.
    train, test = [], []
    for seed in range(20):
        ref = reference_image(seed)
        for radius, label in [(0, False), (5, True), (9, True)]:
            sample = ref if not radius else cv2.GaussianBlur(ref, (radius, radius), 0)
            (train if seed < 12 else test).append((sample, label, seed, radius))
    cutoff = calibrate_threshold([metrics(im)["sharpness"] for im, *_ in train],
                                [label for _, label, *_ in train])
    blur_rows = []
    for sample, label, seed, kernel in test:
        score = metrics(sample)["sharpness"]
        blur_rows.append({"seed": seed, "blur_kernel": kernel,
                          "actual_blur": int(label), "laplacian_variance": round(score, 3),
                          "predicted_blur": int(score < cutoff),
                          "correct": int((score < cutoff) == label)})
    save_csv(output / "iteration2_blur.csv", blur_rows)
    accuracy = np.mean([r["correct"] for r in blur_rows])

    # Iteration 3: correction strengths on controlled darkening of held-out images.
    enhancement_rows = []
    for seed in range(12, 20):
        reference = reference_image(seed)
        for factor in (0.35, 0.55, 0.75):
            dark = darken(reference, factor)
            methods = {"uncorrected": dark,
                       "fixed_gamma_0.8": gamma_correct(dark, 0.8),
                       "fixed_gamma_0.6": gamma_correct(dark, 0.6),
                       "adaptive_gamma": adaptive_gamma(dark)[0],
                       "CLAHE": clahe_enhance(dark)}
            if seed == 12 and factor == 0.35:
                for name, candidate in methods.items():
                    cv2.imwrite(str(output / ("sample_" + name + ".png")), candidate)
                cv2.imwrite(str(output / "sample_reference.png"), reference)
            for name, candidate in methods.items():
                enhancement_rows.append({
                    "seed": seed, "darkening_factor": factor, "method": name,
                    "brightness": round(metrics(candidate)["brightness"], 3),
                    "mae": round(mae(reference, candidate), 3),
                    "psnr_db": round(psnr(reference, candidate), 3)
                })
    save_csv(output / "iteration3_enhancement.csv", enhancement_rows)

    # Iteration 4: weighted ranking stress test on perturbations of identical content.
    # This is a sanity check, not a user-preference agreement measure.
    ranking_rows = []
    for seed in range(12, 20):
        reference = reference_image(seed)
        candidates = [reference, darken(reference, 0.35),
                      cv2.GaussianBlur(reference, (9, 9), 0),
                      low_contrast(reference, 0.3)]
        ranking = rank_photos(candidates)
        for rank, (index, score) in enumerate(ranking, 1):
            ranking_rows.append({"seed": seed, "rank": rank,
                                 "variant": ["reference", "dark", "blurred", "low_contrast"][index],
                                 "score": round(score, 4),
                                 "reference_ranked_first": int(ranking[0][0] == 0)})
    save_csv(output / "iteration4_ranking.csv", ranking_rows)
    top1 = np.mean([r["reference_ranked_first"] for r in ranking_rows])

    summary = (
        "Controlled synthetic evaluation (not real-photo or human-preference results)\n"
        f"Iteration 2 blur threshold (Laplacian variance): {cutoff:.3f}\n"
        f"Iteration 2 held-out accuracy: {accuracy:.1%} ({sum(r['correct'] for r in blur_rows)}/{len(blur_rows)})\n"
        f"Iteration 4 original image top-ranked: {top1:.1%} across 8 synthetic groups\n"
        "Iteration 3: inspect iteration3_enhancement.csv for MAE and PSNR by method.\n"
        "Thresholds and scoring weights require validation on labeled real photos.\n"
    )
    (output / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.output)
