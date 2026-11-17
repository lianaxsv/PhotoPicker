"""Iteration 2: cross-content and multi-resolution exposure comparison.

Requires scikit-image. Execute: python scripts/evaluate_iteration_02.py
The script tests deterministic synthetic transformations, not human-labeled exposure quality.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage import data

from photo_quality.core import analyze_image

DEV = ("astronaut", "coffee", "chelsea", "camera", "rocket")
PREVIOUS = ("hubble_deep_field", "grass", "gravel", "moon", "page", "text",
            "brick", "coins", "horse", "immunohistochemistry", "retina")
ADDITIONAL = ("clock", "checkerboard", "colorwheel", "logo",
              "microaneurysms", "shepp_logan_phantom", "lfw_subset")
SIZES = (128, 256, 512)


def source(name):
    img = getattr(data, name)()
    if name == "lfw_subset":
        img = img[0]
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img = np.uint8(np.clip(img * 255 if img.max() <= 1 else img, 0, 255))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if img.ndim == 2 else cv2.COLOR_RGB2BGR)


def run():
    rows = []
    for cohort, names in [("development", DEV), ("previous_holdout", PREVIOUS),
                          ("additional_content", ADDITIONAL)]:
        for name in names:
            for size in SIZES:
                reference = cv2.resize(source(name), (size, size), interpolation=cv2.INTER_AREA)
                variants = {"reference": reference,
                            "moderate": np.uint8(np.clip(reference.astype("float32") * 1.3 + 10, 0, 255)),
                            "severe": np.uint8(np.clip(reference.astype("float32") * 1.7 + 35, 0, 255)),
                            "extreme": np.uint8(np.clip(reference.astype("float32") * 2 + 60, 0, 255)),
                            "contrast_reduced": np.uint8(np.clip(reference.astype("float32") * .35 + 82, 0, 255))}
                for condition, image in variants.items():
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    mean = float(gray.mean())
                    highlights = float(np.mean(gray >= 243))
                    shadows = float(np.mean(gray <= 12))
                    midtones = float(np.mean((gray > 25) & (gray < 230)))
                    before = int(mean > 205 or highlights > .18)
                    after = int("overexposed" in analyze_image(image)["issues"])
                    predicted_after = int(mean > 205 or
                        (highlights > .18 and not (shadows > .18 and midtones < .08)))
                    assert after == predicted_after, "Unexpected classifier implementation"
                    rows.append({"cohort": cohort, "image": name, "size": size,
                        "condition": condition, "brightness": round(mean, 4),
                        "highlight_fraction": round(highlights, 5),
                        "shadow_fraction": round(shadows, 5),
                        "midtone_fraction": round(midtones, 5),
                        "before_flag": before, "after_flag": after})
    return rows


if __name__ == "__main__":
    results = run()
    destination = Path("results/iteration_02_comparison.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for cohort in ("development", "previous_holdout", "additional_content"):
        print(cohort)
        for condition in ("reference", "moderate", "severe", "extreme", "contrast_reduced"):
            subset = [r for r in results if r["cohort"] == cohort and r["condition"] == condition]
            print(condition, f"{sum(r['before_flag'] for r in subset)}/{len(subset)}",
                  "->", f"{sum(r['after_flag'] for r in subset)}/{len(subset)}")
    print("Saved:", destination)
