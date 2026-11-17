# Iteration 5 — Real-photo content validation (2026-11-17)

## What was actually sourced

We researched Wikimedia Commons photos and their attribution requirements, but direct downloads from Wikimedia Commons and the scikit-image data host failed in this execution environment. **No new photos were scraped or downloaded from the internet** for this run.

Instead, evaluation used nine genuine photographic images distributed locally with scientific Python packages:
- scikit-image: astronaut, coffee, chelsea, camera, rocket, hubble_deep_field, retina, immunohistochemistry.
- Matplotlib: grace_hopper.jpg.

Source documentation: https://scikit-image.org/docs/stable/api/skimage.data.html and https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html.

Not all of these are ordinary consumer photographs: the astronomical and medical images are intentional stress cases. These sources have appeared in previous project iterations; therefore **this is not an unseen generalization set**. Copyright/license terms should be checked per source before redistributing original photos.

## Experimental setup

Each original photo was resized with its aspect ratio preserved to fit inside 420 x 420 pixels. Seven conditions per source: unchanged; darkened by 0.45x and 0.70x; brightened by 1.50x +25; Gaussian blur sigma=2.5; contrast compressed 0.30x around 127; Gaussian noise sigma=18 with deterministic random seed. Thus 9 source images x 7 conditions = 63 **correlated cases**. Each was evaluated unchanged, through the current automatic enhancement, and with Balanced studio strength 80%, totaling 189 image-method measurements.

MAE/PSNR compare to the source photo; lower MAE is better **pixel reconstruction**, not necessarily better perceived appearance. No person rated images; no human preference accuracy was measured.

## Mean MAE by condition (nine source photos each)

| Condition | Input | Auto | Balanced 80% |
| --- | ---: | ---: | ---: |
| Unchanged | 0.00 | 16.38 | 19.31 |
| Dark x0.45 | 53.79 | 30.29 | 21.02 |
| Dark x0.70 | 29.52 | 35.13 | 19.42 |
| Bright x1.50 +25 | 56.18 | 57.21 | 50.43 |
| Gaussian blur sigma 2.5 | 7.18 | 20.66 | 21.21 |
| Contrast compression | 48.54 | 44.12 | 45.33 |
| Gaussian noise sigma 18 | 13.45 | 29.72 | 29.00 |

## Outcome counts (number of nine cases per condition with lower/higher MAE than uncorrected)

| Condition | Auto improves | Auto worsens | Balanced improves | Balanced worsens |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 0 | 3 | 0 | 9 |
| Dark x0.45 | 7 | 2 | 9 | 0 |
| Dark x0.70 | 4 | 3 | 7 | 2 |
| Bright x1.50 +25 | 3 | 1 | 7 | 2 |
| Blur sigma 2.5 | 0 | 3 | 0 | 9 |
| Contrast compression | 7 | 2 | 6 | 3 |
| Noise sigma 18 | 0 | 3 | 0 | 9 |

Zero changes are neither improvements nor degradations; they are not shown in the table.

## Detector flags

Flags count images matching a *heuristic*, not reference-validated defect labels.

| Synthetic condition | Underexposed | Overexposed | Possibly blurry | Low contrast |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 2/9 | 0/9 | 0/9 | 3/9 |
| Dark x0.45 | 9/9 | 0/9 | 2/9 | 9/9 |
| Dark x0.70 | 5/9 | 0/9 | 1/9 | 5/9 |
| Bright x1.50 +25 | 1/9 | 3/9 | 0/9 | 2/9 |
| Blur sigma 2.5 | 2/9 | 0/9 | 9/9 | 3/9 |
| Contrast compression | 0/9 | 0/9 | 6/9 | 9/9 |
| Noise sigma 18 | 2/9 | 0/9 | 0/9 | 3/9 |

## What to conclude

- Automatic enhancement can improve strong synthetic underexposure, but makes other conditions worse by reference MAE, especially noise/blur.
- The Balanced preset often fixes darkening more effectively, but overprocesses all nine originally clean images (nonzero MAE).
- Both methods lack actual denoising/deblurring; changing apparent sharpness is not reconstructing lost detail.
- A naturally dark image can be flagged despite correctly representing the scene.
- These reference images are already used in development, and we have **no human labels**, so this is not broad real-photo generalization evidence.
- No production changes are justified by this benchmark alone. Request/use fresh permissioned camera images with human-defined desirable exposure and ranked preferences before further tuning.

## Reproduction and history

Run `python -m pip install -r requirements-dev.txt scikit-image matplotlib`; then `python scripts/evaluate_real_photos_iteration_05.py`. Output: `results/iteration_05/metrics.csv`, `results/iteration_05/flags.csv`.

This experiment was run against the latest GitHub-verified detection logic and current enhancement methods, using a local execution copy. No PhotoPick production algorithm was modified.

Commits:
- `24fdab2` — Reproducible real-photo test script
- This commit — report and explicit limitations

Full report append: *PhotoPick_Test_Report_Iterations_1_to_5.pdf* (local artifact, not tracked in GitHub).
