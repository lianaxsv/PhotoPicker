"""Reproduce the five-image overexposure sensitivity test.

Run: python scripts/evaluate_exposure_iteration.py --output results/exposure_iteration_01.csv
Requires scikit-image (development dependency).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage import data
from photo_quality.core import analyze_image

REFERENCE_NAMES = ('astronaut', 'coffee', 'chelsea', 'camera', 'rocket')

def run():
    rows = []
    for name in REFERENCE_NAMES:
        raw = getattr(data, name)()
        raw = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        original = (cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.ndim == 3
                    else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR))
        modified = np.clip(original.astype(np.float32) * 1.7 + 35, 0, 255).astype(np.uint8)
        for label, img in (('reference', original), ('brightened', modified)):
            metrics = analyze_image(img)
            rows.append({'image': name, 'condition': label,
                         'mean_brightness': metrics['brightness'],
                         'highlight_fraction': metrics['highlight_fraction'],
                         'overexposed': int('overexposed' in metrics['issues']),
                         'issues': '|'.join(metrics['issues'])})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('results/exposure_iteration_01.csv'))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = run()
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for condition in ('reference', 'brightened'):
        group = [r for r in rows if r['condition'] == condition]
        print(condition, f"flagged {sum(r['overexposed'] for r in group)}/{len(group)}")
    print('Wrote', args.output)

if __name__ == '__main__':
    main()
