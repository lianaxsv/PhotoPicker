# Iteration 7 — Richardson–Lucy deconvolution experiment (2026-11-17)

**Status:** Experiment only. Production app unchanged. Awaiting human visual ratings.

## Motivation
One evaluator rejected all eight current blur-enhancement examples; increasing local contrast did not restore detail. This is one person's qualitative preference, not a population estimate.

## Method
Richardson–Lucy iterative deconvolution (scikit-image) with a **known point-spread function (PSF)**. The evaluation uses the eight previously generated controlled-blur samples (coffee, cat, astronaut, rocket, camera, Grace Hopper, grass, Hubble), with four samples in the development group and four additional samples in the secondary group. PSF parameters match the synthetic degradation (motion lengths 15–25 px and angles, or Gaussian defocus sigma 3.2–4.5). Reconstructed images were clipped to 8-bit values, with reflect-padded boundaries. Tested 8 and 15 deconvolution iterations. This is a favorable **oracle PSF** setup and is not proof that blind restoration works on phone photos.

Current Auto and Balanced 80% outputs were reused from the same comparisons, with JPEG artifacts carried through. Reconstruction metrics compare against each sample's available sharp source image. The reference alignment is appropriate for these deliberately degraded photos; these are *not* real independently captured blurry/sharp pairs.

## Measured mean results

| Cohort | Method | Mean MAE (lower better) | Mean PSNR dB (higher better) |
|---|---|---:|---:|
| 4 development photos | Blurry input | 9.89 | 24.06 |
| 4 development photos | Current Auto | 26.28 | 18.90 |
| 4 development photos | Balanced 80% | 23.34 | 19.41 |
| 4 development photos | RL, 8 iterations | **8.74** | **25.00** |
| 4 development photos | RL, 15 iterations | 9.02 | 24.65 |
| 4 secondary photos | Blurry input | 13.57 | 21.39 |
| 4 secondary photos | Current Auto | 26.11 | 18.26 |
| 4 secondary photos | Balanced 80% | 22.81 | 18.84 |
| 4 secondary photos | RL, 8 iterations | **12.36** | 22.77 |
| 4 secondary photos | RL, 15 iterations | 12.11 | **23.13** |

RL8 improved reference MAE vs blurry input for 8/8 examples. Mean combined MAE across eight sources: input 11.73, RL8 10.42. RL15 not universally preferable.

## Critical limitations
- **Oracle information:** Blur kernel type, direction, and width were known because we generated the blur. These would need to be estimated for authentic motion-blurred photographs.
- Only eight correlated examples, mostly reused from earlier tests, not independent held-out validation.
- Lower MAE is not evidence of better human-perceived quality; ringing and deconvolution artifacts still possible.
- No improvement guarantee for severe nonuniform motion blur, subject motion, depth variation, or saturated highlights.
- The user's personal photo has not been committed or included in the repository; no real-photo deconvolution claim is made.
- Human ratings are pending. Do not deploy this candidate until real-photo evaluations support it.

## Git history
- `192898c` — self-contained reproducible evaluation script (requires local comparison images, not stored in repo)
- This documentation commit — numeric outcomes, caveats, and decision.

## Reproduction
```bash
pip install -r requirements-dev.txt scikit-image
python scripts/evaluate_deconvolution_iteration_07.py /path/to/blurry_comparison_images results/deconvolution_07.csv
```

Inputs are the previously generated `*_before.jpg`, `*_auto.jpg`, `*_balanced.jpg`, and `*_sharp_reference.jpg` outputs. Their raw photographs are not redistributed in the repository.

**Next:** rate deconvolution outputs blind, then evaluate PSF estimation on genuine real-camera blur (e.g. the licensed Kwentar dataset) without tuning per test image.
