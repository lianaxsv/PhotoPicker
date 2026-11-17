# CIS 5810 — Iteration notes and evaluation

PhotoPick is an explainable image quality detector and enhancer.

## Original proposal: controlled gamma experiment

| Image | Mean brightness | MAE to reference |
| --- | ---: | ---: |
| Original reference | 115.27 | 0.00 |
| Darkened image | 52.96 | 61.64 |
| Fixed gamma 0.8 | 72.40 | 42.98 |
| Fixed gamma 0.6 | 98.46 | 16.52 |

These are preliminary figures from the group proposal, based on its controlled illustration, **not** the demo gallery.

## Iteration plan

1. Baseline: Laplacian variance for sharpness, grayscale mean for exposure, intensity standard deviation for contrast; fixed gamma = 0.8.
2. Gamma strength: compare 0.8 and 0.6 on the same reference image; stronger correction reduces error for the tested darkened illustration.
3. Detection calibration: assess false positives from low-texture scenes, naturally dark photos, and spatially varying contrast. Use labeled photos to tune thresholds and measure precision/recall.
4. Conditional enhancement: compare fixed gamma, adaptive gamma, CLAHE, and selective sharpening. Apply only needed corrections to avoid artifacts.
5. Ranking: normalize sharpness/exposure/contrast, then compare heuristic weights against human judgments.

## Evaluation procedure

- Use the same source images across methods, including a held-out validation set.
- Generate controlled degradations (different exposure strengths, blur radii, contrast compression).
- Record MAE and PSNR against the original images.
- Report accuracy/precision/recall on manually labeled blur, exposure, and contrast examples.
- Collect reference rankings from human participants for top-photo and pairwise agreement.
- Include examples of negative outcomes, overcorrection, severe blur, and false positives.

## Important reporting note

The later numerical tables in the discussion journal were drafted as illustrative examples; they do not constitute verified experimental outcomes. AI-generated comparison images are likewise illustrations, not direct outputs corresponding to the proposal's numeric measurements. Use the evaluation script and saved test outputs before claiming new measured results.
