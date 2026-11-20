# Iteration 8 — Blur and sharpness threshold calibration (Michael Ren)

**November 19, 2026. Evaluation only; no production changes.**

## Motivation and pipeline

PhotoPick's existing `photo_quality/core.py` flags possible blur when Laplacian variance is below **75**. We evaluated the same detector across resolution, texture, brightness, and blur type. This work covers Michael's assigned contributions: blur and sharpness analysis, dataset preparation, testing, and threshold calibration. It does not replace the app, enhancement algorithms, or the existing **40% sharpness, 35% exposure, 25% contrast** ranking rule.

We used 12 packaged scikit-image source images, split by image source: six for calibration (astronaut, coffee, chelsea, camera, rocket, grass) and six for evaluation (coins, brick, gravel, moon, page, hubble_deep_field). Each was resized preserving aspect ratio to maximum sides of 128, 256, and 512 pixels. We tested unchanged, darkened by 0.55, contrast reduced by 0.55, Gaussian blur sigma 3, and horizontal 15-pixel motion blur. That is **180 correlated conditions from 12 source images** (90 per cohort). Some source photos have been used in earlier project iterations, so this is not a globally independent test. The labels indicate applied transformations, not human-annotated photo quality.

For threshold calibration, we maximized balanced accuracy (average sensitivity and specificity) **only on the six development sources**; tie-breaking favors the original default. We applied the selected cutoff to the separate six-source evaluation set, without retuning.

## Recorded results from local execution

**Candidate threshold 110.895** versus production default **75.000**.

| Source cohort | Cutoff | TP | TN | FP | FN | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development, 90 conditions | 75.000 | 29 | 52 | 2 | 7 | 80.6% | 96.3% | 88.4% |
| Development, 90 conditions | 110.895 | 32 | 52 | 2 | 4 | 88.9% | 96.3% | 92.6% |
| Evaluation, 90 conditions | 75.000 | 25 | 45 | 9 | 11 | 69.4% | 83.3% | 76.4% |
| Evaluation, 90 conditions | 110.895 | 28 | 43 | 11 | 8 | 77.8% | 79.6% | 78.7% |

Evaluation-group balanced accuracy by image size:

| Maximum side | Current 75 | Candidate 110.895 |
|---|---:|---:|
| 128 pixels | 73.6% | 77.8% |
| 256 pixels | 77.8% | 86.1% |
| 512 pixels | 77.8% | 72.2% |

For the 18 injected Gaussian-blur evaluation conditions, **18/18** were flagged at either threshold. For the 18 motion-blur conditions, **7/18** were flagged at 75 versus **10/18** at the candidate threshold. Of the 54 negative-control conditions, **9** were flagged at 75 and **11** at 110.895. On the low-texture moon source, there were seven flagged negative controls. Because the original images are not human-confirmed sharp, these are synthetic-label discrepancies rather than proven perceptual errors.

## Findings and next steps

Increasing the threshold caught more synthetic motion blur but produced more false positives. The candidate's held-out improvement was **2.3 percentage points** in balanced accuracy and reversed direction at 512 pixels. Image content and resolution affect Laplacian variance, and reductions in brightness or contrast may trigger the blur flag even when no blur is injected. **Retain the team's existing production cutoff of 75**, pending more data. Next evaluate real, independently collected, human-labeled motion-blurred and defocused photos, split by source scene, and compare a resolution-aware threshold while tracking false alarms.

This study evaluates **detection**, not restoration; it does not conflict with the existing team's deconvolution or human-preference iterations. Multiple images generated from one source are correlated, and MAE or synthetic transformation labels do not establish subjective photo quality.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt scikit-image
pytest -q tests/test_blur_threshold_michael.py
python scripts/evaluate_blur_threshold_michael.py --output results/blur_threshold_michael
```

Outputs in the ignored `results/blur_threshold_michael` directory: `case_metrics.csv` (every measured condition) and `summary.md` (confusion matrices by cohort and size). This study introduces a new script and tests only; all teammate files remain unchanged.
