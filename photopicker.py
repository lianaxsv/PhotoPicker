"""Image-quality metrics, simple threshold calibration, enhancement and ranking."""
from dataclasses import dataclass
import cv2
import numpy as np

WEIGHTS = {"sharpness": 0.45, "exposure": 0.35, "contrast": 0.20}


def as_bgr(image):
    if image is None or image.size == 0:
        raise ValueError("Expected a nonempty image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected grayscale or BGR image")
    return image


def metrics(image):
    gray = cv2.cvtColor(as_bgr(image), cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": brightness,
        "contrast": float(gray.std()),
        "dark_fraction": float(np.mean(gray < 30)),
        "bright_fraction": float(np.mean(gray > 225)),
    }


@dataclass
class Thresholds:
    blur: float = 80.0
    min_brightness: float = 75.0
    max_brightness: float = 190.0
    min_contrast: float = 35.0


def detect(image, thresholds=None):
    t = thresholds or Thresholds()
    m = metrics(image)
    return {
        "blur": m["sharpness"] < t.blur,
        "underexposed": m["brightness"] < t.min_brightness,
        "overexposed": m["brightness"] > t.max_brightness,
        "low_contrast": m["contrast"] < t.min_contrast,
    }


def calibrate_threshold(values, labels):
    """Choose a threshold separating low-metric (positive) from high-metric images.

    Calibrate using only training examples; report accuracy on separate examples.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if len(values) != len(labels) or not len(values):
        raise ValueError("values and labels must be nonempty, same-length arrays")
    options = np.unique(values)
    candidates = np.r_[options[0] - 1, (options[:-1] + options[1:]) / 2, options[-1] + 1]
    scores = [float(np.mean((values < v) == labels)) for v in candidates]
    return float(candidates[int(np.argmax(scores))])


def gamma_correct(image, gamma=0.8):
    """Gamma < 1 brightens under the power-law convention used in our proposal."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    lut = np.clip(255 * ((np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(as_bgr(image), lut)


def adaptive_gamma(image, target=115.0, min_gamma=0.35, max_gamma=1.8):
    mean = max(metrics(image)["brightness"], 1.0)
    target = float(np.clip(target, 1, 254))
    # For constant-intensity approximation: (mean/255)^gamma = target/255
    gamma = np.log(target / 255) / np.log(np.clip(mean / 255, 1e-5, 0.99999))
    gamma = float(np.clip(gamma, min_gamma, max_gamma))
    return gamma_correct(image, gamma), gamma


def clahe_enhance(image, clip_limit=2.0):
    lab = cv2.cvtColor(as_bgr(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2BGR)


def enhance(image, thresholds=None):
    """Apply conditional enhancements; severe blur is only flagged."""
    image = as_bgr(image)
    flags = detect(image, thresholds)
    corrected = image.copy()
    methods = []
    if flags["underexposed"] or flags["overexposed"]:
        corrected, gamma = adaptive_gamma(corrected)
        methods.append("adaptive_gamma={:.3f}".format(gamma))
    if flags["low_contrast"]:
        corrected = clahe_enhance(corrected)
        methods.append("CLAHE")
    return corrected, flags, methods


def quality_score(image, sharpness_scale=250.0):
    """Heuristic score in [0, 1]; weights are hypotheses, not validated preferences."""
    m = metrics(image)
    sharpness = float(np.clip(m["sharpness"] / sharpness_scale, 0, 1))
    exposure = float(np.clip(1 - abs(m["brightness"] - 127.5) / 127.5, 0, 1))
    contrast = float(np.clip(m["contrast"] / 64.0, 0, 1))
    return (WEIGHTS["sharpness"] * sharpness +
            WEIGHTS["exposure"] * exposure +
            WEIGHTS["contrast"] * contrast)


def rank_photos(images):
    """Return (index, score) pairs in descending order; original order breaks ties."""
    return sorted(enumerate(quality_score(im) for im in images),
                  key=lambda pair: (-pair[1], pair[0]))
