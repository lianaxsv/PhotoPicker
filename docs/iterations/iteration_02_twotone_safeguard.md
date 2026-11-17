# Iteration 2 — Two-tone overexposure safeguard (November 17, 2026)

## Problem and rationale

Iteration 1 flagged a high-contrast black/white horse silhouette as overexposed. Mean grayscale brightness was about 170.6, but roughly 65% of grayscale pixels were near-white; the 18% highlight-clipping shortcut therefore classified an intentionally two-tone image as defective.

This iteration tests whether distinguishing near-binary imagery from ordinary photographs can suppress that class of false positive without lowering severe-brightening detection on the previously examined image set. This is a targeted heuristic, not semantic scene understanding.

## Change

Maintain exposure labeling when mean grayscale brightness >205. For the highlight-clipping path only, require clipped fraction >0.18 and **do not use that shortcut** when all of these hold: shadow fraction (>0.18, intensity <=12), clipped fraction (>0.18, intensity >=243), and midtone fraction (<0.08, intensities 26–229).

Importantly, the independent `highlight_clipping` flag remains present; only the `overexposed` label is suppressed in near-binary cases.

## Separate commits

- `8329d65` — regression tests (created before the fix; intended to fail on the prior algorithm)
- `17f4a67` — implementation of the two-tone safeguard
- `470e58c` — reproducible cross-cohort benchmark

## Measured comparison

All counts are from deterministic synthetic transformations on images bundled with scikit-image. Three resized resolutions per source (128, 256, and 512 pixels) are **correlated measurements**, not independent photographs.

### Prior 11-source set (33 source-size cases per condition)

| Condition | Before | After |
|---|---:|---:|
| Unchanged reference flagged | 3/33 | 0/33 |
| Moderate brightening flagged | 9/33 | 6/33 |
| Severe brightening flagged | 27/33 | 27/33 |
| Extreme brightening flagged | 27/33 | 27/33 |
| Contrast reduction flagged | 0/33 | 0/33 |

This set was **previously inspected to identify the horse-image failure**, so it is not an untouched generalization test for this iteration. The reduction in moderate-brightening flags is a sensitivity trade-off, not necessarily an error reduction.

### Seven additional image sources (21 source-size cases per condition)

Images: clock, checkerboard, colorwheel, logo, microaneurysms, shepp_logan_phantom, and one lfw_subset face thumbnail.

| Condition | Before | After |
|---|---:|---:|
| Unchanged reference flagged | 6/21 | 6/21 |
| Moderate brightening flagged | 6/21 | 5/21 |
| Severe brightening flagged | 12/21 | 12/21 |
| Extreme brightening flagged | 15/21 | 15/21 |
| Contrast reduction flagged | 0/21 | 0/21 |

The unchanged checkerboard and logo still trigger the heuristic (at all three resolutions). **The new rule does not generalize reliably to all graphical or white-background images.** These source types are not representative of camera photographs.

### Development reference images (5 sources x 3 sizes)

Unchanged 0/15 -> 0/15, moderate 6/15 -> 6/15, severe 12/15 -> 12/15, extreme 12/15 -> 12/15, contrast reduction 0/15 -> 0/15.

## Automated validation

The existing seven tests, two Iteration 1 regression tests, and four Iteration 2 regression tests all pass: **13/13** locally. This establishes execution and specified behavior on those test cases, not overall photographic detection accuracy. Live Streamlit UI behavior was not browser-tested here.

## Limitations and next steps

- An image can be both very dark and clipped, or may be deliberately overexposed for artistic reasons.
- Arbitrary brightening is not the same as a human-labeled bad exposure.
- A few synthetic source types dominate this tiny sample, and resized versions are correlated.
- Thresholds were chosen after examining the known false positive, introducing development-set bias.
- Need a truly untouched, real-camera photo dataset with independent human labels to evaluate precision, recall, and false positives. Do not tune further on the previously examined holdout.
- Next priority should be ranking calibration, preserving failed cases in the audit trail.

## Reproduction

```bash
pip install -r requirements-dev.txt scikit-image
pytest -q
python scripts/evaluate_iteration_02.py
```

Full case results are written to `results/iteration_02_comparison.csv`. The output is a local file, not automatically committed to Git.
