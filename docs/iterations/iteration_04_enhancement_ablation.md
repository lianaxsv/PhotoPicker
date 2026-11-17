# Iteration 4 — Automatic enhancement ablation (Nov 17, 2026)

## Objective

Determine whether the current automatic enhancement pipeline actually improves reconstruction quality on different input types. Earlier testing showed that gamma/CLAHE can overprocess already-clean, blurred, or noisy images.

## Candidate (not merged)

A conservative rule was compared against the unchanged production pipeline: only apply adaptive gamma when mean grayscale brightness <75 and underexposure is flagged; otherwise return the original unchanged. This candidate deliberately skips contrast correction and sharpening. **The candidate was NOT shipped**, because it worsened overall error on the 11-source secondary set.

## Evaluation methodology

16 image sources in two cohorts: 5 development sources and 11 secondary sources bundled with scikit-image. Each source resized to 256x256; evaluated with 11 variants: clean, five degrees of synthetic darkening (0.25, 0.40, 0.55, 0.70, 0.85), Gaussian blur sigma 2/5, synthetic brightening, contrast compression, and Gaussian additive noise sigma 18. There were 176 condition/source measurements, **not 176 independent photographs**.

For each variant, mean absolute error (MAE) to the source reference was measured before correction, after the existing auto pipeline, and after the proposed safeguard. Lower is better. The untouched references are only convenient pixel targets, not necessarily the human-preferred look.

## Actual measured outcomes

| Cohort | Cases | Input mean MAE | Existing auto mean MAE | Candidate mean MAE | Auto worsens vs input | Candidate worsens vs input |
|---|---:|---:|---:|---:|---:|---:|
| 5 development sources | 55 | 36.08 | 31.54 | 31.70 | 12/55 | 7/55 |
| 11 secondary sources | 121 | 39.27 | 28.91 | 31.29 | 33/121 | 10/121 |

Across 11 secondary sources, the proposed guard lowered the number of harmful operations, but increased average reconstruction error from 28.91 to 31.29. Thus it is **not a demonstrated overall improvement**. Existing automatic enhancement also performs imperfectly (33/121 harmful cases relative to doing nothing).

Examples from the secondary cohort (mean MAE):
- Clean input: existing auto 14.19, candidate 4.95 (both should ideally be 0).
- Blur sigma 2: existing auto 18.02, candidate 14.88.
- Noise sigma 18: existing auto 25.98, candidate 16.95.
- Darkened x0.25: existing auto 26.94, candidate 35.48 (candidate worse).
- Brightened: existing auto 68.95, candidate 85.62 (candidate worse).

The candidate also modified one naturally dark clean reference. This highlights why mean brightness is not enough to decide whether an image needs correction.

## Conclusion and next action

**Do not deploy the candidate.** Changing the classifier alone cannot reliably distinguish intentionally dark scenes from exposure defects, and the candidate sacrifices correction quality for some defect categories. Production `photo_quality/core.py` remains unchanged in this iteration.

Recommended next approach: assemble a genuinely independent set of camera photographs with labels and a reference/fidelity target, separate development/test groups by source, then develop targeted filter gating using measured defect severity and reference-based evaluation. Avoid tuning to the 16 scikit-image examples.

## Git history

- `0cfe10d`: reproducible evaluation script, with no production-code changes
- This separate documentation commit records negative as well as positive findings

Run: `python scripts/evaluate_iteration_04.py` after installing scikit-image. Output: `results/iteration_04.csv`.
