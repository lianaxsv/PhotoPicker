"""Michael Ren: blur/sharpness robustness for the existing PhotoPick analyzer.

Evaluation only: no changes to production threshold, app, or ranking.
Run from repository root after installing scikit-image:
python scripts/evaluate_blur_threshold_michael.py --output results/blur_threshold_michael
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from photo_quality.core import DEFAULT_THRESHOLDS, analyze_image

# By source photo, not by variant; reused elsewhere in the team's project.
DEVELOPMENT_SOURCES = ('astronaut', 'coffee', 'chelsea', 'camera', 'rocket', 'grass')
EVALUATION_SOURCES = ('coins', 'brick', 'gravel', 'moon', 'page', 'hubble_deep_field')
SIZES = (128, 256, 512)


def scaled_photo(img: np.ndarray, max_side: int) -> np.ndarray:
    """Preserve aspect ratio and convert packaged samples into BGR uint8."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.uint8(img)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def motion_blur(image: np.ndarray, length: int = 15) -> np.ndarray:
    """Simulate one-direction horizontal camera motion."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def degraded_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    f = image.astype(np.float32)
    return {
        'original': image,
        'dark_x0.55': np.clip(f * 0.55, 0, 255).astype(np.uint8),
        'contrast_x0.55': np.clip((f - 127.5) * 0.55 + 127.5, 0, 255).astype(np.uint8),
        'gaussian_sigma3': cv2.GaussianBlur(image, (0, 0), 3.0),
        'motion_15px': motion_blur(image),
    }


def balanced_accuracy(labels, predictions) -> float:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.size == 0 or not np.any(labels) or np.all(labels):
        raise ValueError('Balanced accuracy requires positive and negative examples')
    return float((np.mean(predictions[labels]) + np.mean(~predictions[~labels])) / 2)


def choose_threshold(scores, labels, default: float) -> tuple[float, float]:
    """Tune on development data only; resolve ties nearest existing cutoff."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.size == 0 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Scores and labels must be matching, finite nonempty arrays')
    unique = np.unique(scores)
    candidates = np.unique(np.r_[0.0, (unique[:-1] + unique[1:]) / 2,
                                 np.nextafter(unique[-1], np.inf), default])
    outcomes = [balanced_accuracy(labels, scores < t) for t in candidates]
    best = max(outcomes)
    tied = [float(t) for t, score in zip(candidates, outcomes)
            if abs(score - best) < 1e-12]
    return min(tied, key=lambda t: (abs(t - default), t)), best


def confusion_stats(rows, prediction_key: str) -> dict:
    tp = sum(r['applied_blur'] == 1 and r[prediction_key] == 1 for r in rows)
    tn = sum(r['applied_blur'] == 0 and r[prediction_key] == 0 for r in rows)
    fp = sum(r['applied_blur'] == 0 and r[prediction_key] == 1 for r in rows)
    fn = sum(r['applied_blur'] == 1 and r[prediction_key] == 0 for r in rows)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {'n': len(rows), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'sensitivity': recall, 'specificity': specificity,
            'balanced_accuracy': (recall + specificity) / 2
            if recall is not None and specificity is not None else None}


def collect_rows() -> list[dict]:
    try:
        from skimage import data
    except ImportError as exc:
        raise RuntimeError('Install scikit-image: pip install scikit-image') from exc
    rows = []
    for cohort, names in [('development', DEVELOPMENT_SOURCES),
                          ('evaluation', EVALUATION_SOURCES)]:
        for name in names:
            original = getattr(data, name)()
            for size in SIZES:
                image = scaled_photo(original, size)
                for variant, candidate in degraded_variants(image).items():
                    analysis = analyze_image(candidate)
                    rows.append({
                        'cohort': cohort, 'source': name, 'max_side_px': size,
                        'variant': variant,
                        'applied_blur': int(variant in ('gaussian_sigma3', 'motion_15px')),
                        'laplacian_variance': analysis['sharpness'],
                        'default_predicted_blur': int('possibly_blurry' in analysis['issues']),
                        'calibrated_predicted_blur': 0,
                    })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value) -> str:
    return 'n/a' if value is None else f'{value:.1%}'


def report(rows: list[dict], threshold: float, train_score: float) -> str:
    default = float(DEFAULT_THRESHOLDS['blur'])
    header = (
        '# Blur threshold study — Michael Ren\n\n'
        'Evaluation only, using the existing photo_quality.core.analyze_image '
        'and its unmodified production blur threshold.\n\n'
        f'- Default threshold: **{default:.3f}** Laplacian variance\n'
        f'- Development-calibrated candidate: **{threshold:.3f}** '
        f'(development balanced accuracy {fmt_pct(train_score)})\n'
        '- Data: 12 package-bundled source images; 6 development and 6 evaluation. '
        'Three resolutions and five transformations per source.\n'
        '- Ground truth: artificially injected Gaussian or motion blur versus '
        'original, darkened, and contrast-reduced conditions. '
        'This is NOT subjective blur labeling or general real-camera accuracy.\n'
        '- Evaluation sources are disjoint within this experiment, '
        'but some appeared in earlier project iterations.\n\n'
        '| Cohort / resolution | Rule | N | TP | TN | FP | FN | Sensitivity | Specificity | Balanced accuracy |\n'
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    )
    lines = []
    for cohort in ('development', 'evaluation'):
        for size in ('all',) + SIZES:
            subset = [r for r in rows if r['cohort'] == cohort and
                      (size == 'all' or r['max_side_px'] == size)]
            for kind, key in [('Current', 'default_predicted_blur'),
                              ('Calibrated', 'calibrated_predicted_blur')]:
                m = confusion_stats(subset, key)
                lines.append(f'| {cohort} / {size} | {kind} | {m["n"]} | {m["tp"]} | '
                             f'{m["tn"]} | {m["fp"]} | {m["fn"]} | '
                             f'{fmt_pct(m["sensitivity"])} | {fmt_pct(m["specificity"])} | '
                             f'{fmt_pct(m["balanced_accuracy"])} |')
    return header + '\n'.join(lines) + '\n\n'


def main(output: Path):
    rows = collect_rows()
    default = float(DEFAULT_THRESHOLDS['blur'])
    training = [r for r in rows if r['cohort'] == 'development']
    threshold, train_score = choose_threshold(
        [r['laplacian_variance'] for r in training],
        [r['applied_blur'] for r in training], default)
    for r in rows:
        r['calibrated_predicted_blur'] = int(r['laplacian_variance'] < threshold)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'case_metrics.csv', rows)
    summary = report(rows, threshold, train_score)
    (output / 'summary.md').write_text(summary, encoding='utf-8')
    print(summary)
    return rows, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('results/blur_threshold_michael'))
    main(parser.parse_args().output)
