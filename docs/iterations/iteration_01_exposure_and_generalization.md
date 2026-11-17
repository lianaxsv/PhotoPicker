# Iteration 1: Overexposure detection and held-out robustness (2026-11-17)

## Baseline failure

The original overexposure label checked only whether grayscale mean brightness exceeded 205. In the five-image controlled brightening audit (astronaut, coffee, chelsea, camera, rocket), this flagged 1/5 brightened inputs and 0/5 unchanged references.

## Hypothesis and focused implementation

Heavy highlight clipping can indicate overexposure even if the average intensity is below 205. The revised rule flags overexposure when mean grayscale brightness >205 **OR** more than 18% of grayscale pixels are >=243. This is a simple heuristic, not a proven universal exposure detector.

## Separate commits

- `f1b2596`: failing regression example (30% white pixels, moderate mean)
- `68f55ba`: reproducible five-image benchmark
- `f8cdf83`: focused overexposure rule change
- `2ef63f7`: 11-source, 3-resolution held-out evaluation script

## Original five-image evaluation

| Metric | Before | After |
|---|---:|---:|
| Brightened inputs flagged | 1/5 | 4/5 |
| Unchanged references incorrectly flagged | 0/5 | 0/5 |
| Automatic enhancement MAE, brightened inputs (mean) | 72.38 | 60.22 |

The brightened rocket image remained unflagged because only 2.77% of grayscale pixels were above the clipping threshold. These five examples are the development set, not independent generalization evidence.

## Held-out evaluation

Eleven additional scikit-image sources: hubble_deep_field, grass, gravel, moon, page, text, brick, coins, horse, immunohistochemistry, retina. Each resized to 128, 256, and 512 px; five conditions per size (unchanged, moderate brightening, severe brightening, extreme brightening, contrast reduction). **165 condition/size measurements, but only 11 distinct source images**; resolutions are correlated rather than independent samples.

| Condition | Old rule flags | New rule flags |
|---|---:|---:|
| Unchanged reference | 0/33 | 3/33 |
| Moderate brightening | 6/33 | 9/33 |
| Severe brightening | 21/33 | 27/33 |
| Extreme brightening | 24/33 | 27/33 |
| Contrast reduction | 0/33 | 0/33 |

### Failure cases
- The unchanged high-contrast black/white horse image was incorrectly flagged at all three resolutions (near-white fraction ~65%).
- Very dark Hubble and retina images remained unflagged even with severe synthetic brightening.
- Moderate brightening is not inherently a defective exposure. The counts above measure detection of an artificial transformation, **not human-labeled exposure accuracy**.

## Reproduction

```bash
python -m pip install scikit-image opencv-python-headless numpy
python scripts/evaluate_exposure_iteration.py --output results/exposure_iteration_01.csv
python scripts/evaluate_generalization.py --output results/heldout.csv
pytest -q
```

## What these tests do not establish

Not a real-world camera-photo benchmark, not a representative sample of all photographic content, and not an independent estimate of human exposure-label precision/recall. No internet photos were downloaded for these experiments. Keep the held-out set unused for threshold tuning and obtain a third separate test set after any further calibration.

Full report: PhotoPick Testing Report (Arial, black-and-white), sections 10-12, with raw held-out cases in `heldout.csv`.
