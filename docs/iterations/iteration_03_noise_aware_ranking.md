# Iteration 3 - Noise-aware ranking (November 17, 2026)

## Defect and reason to revisit ranking

PhotoPick ranks photographs using exposure, global contrast and raw Laplacian-variance sharpness. Random pixel noise increases Laplacian variance and can therefore increase ranking scores even when it reduces reference-image fidelity. This iteration targets that specific pathology. It does **not** claim to solve all aesthetic ranking errors.

## Change and separate commits

- `aa4002a` - add noise-ranking regression tests
- `99dba50` - rank with Gaussian-smoothed Laplacian variance and a median-filter residual noise proxy; leave the raw sharpness metric for blur detection unchanged
- `23f5433` - add deterministic before-and-after scoring benchmark

The ranking equation retains exposure weight 0.35 and contrast weight 0.25. Its 0.40 sharpness contribution now uses variance of the Laplacian after Gaussian smoothing (sigma=2), and it subtracts `median(abs(gray - medianFilter3x3(gray)))/255` from the normalized weighted score. These are heuristic values and **were chosen after inspecting the development images**, so the development set is not an independent test.

## Measurement setup

Five development reference images, 11 images previously used to evaluate exposure, and 7 additional graphic/microscopy/face examples from scikit-image (23 distinct sources total). Each source is resized to 256x256. Gaussian pixel noise is injected at sigma=8, 16, and 28 with seed 5810, producing 69 paired comparisons. The target is whether the noisy image scores higher than the unmodified reference under the *same* scoring rule. This is a synthetic-fidelity test, not a human photo-preference assessment. Some image groups contain graphics and scientific imagery, not camera photographs.

| Group | Baseline noisy copies ranked higher | After fix |
|---|---:|---:|
| 5 development images (15 noise variants) | 9/15 | 0/15 |
| 11 prior exposure images (33 variants) | 22/33 | 6/33 |
| 7 additional-content images (21 variants) | 20/21 | 13/21 |
| **Total (69 paired comparisons)** | **51/69** | **19/69** |

Thus 32 fewer noise variants ranked above their clean reference. These 69 observations share only 23 independent source images, and some sources were already used in earlier development or testing.

## Checks and limitations

- Local automated suite: 16/16 passed (13 prior plus 3 new regression tests).
- The previous five-group ranking test remains 4/5 clean-reference wins. The **rocket image still ranks below its brightened version**. Increased brightness might genuinely improve an underexposed reference, so this cannot be labeled a definite aesthetic error without human preferences.
- The original still wins only 15/23 synthetic distortion sets overall, unchanged by this noise-specific fix.
- Textured scenes naturally produce high median-filter residuals. The noise proxy may penalize genuine high-frequency detail.
- The additional-content cohort remains weak (13/21 noise copies ranked above clean). No claim of broad generalization is justified.
- The photo gallery now uses the new scoring heuristic. The score scale changes across the iteration; compare paired ranking decisions rather than absolute scores.
- This is not an independent camera-photo test, nor was Streamlit browser UI tested as part of this iteration.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt scikit-image
pytest -q
python scripts/evaluate_ranking_iteration_03.py
```

The script writes `results/ranking_iteration_03.csv`. The full evaluation should be supplemented with a newly collected and human-rated camera dataset before final claims.
