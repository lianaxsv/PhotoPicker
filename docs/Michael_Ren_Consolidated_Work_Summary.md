# Michael Ren — Consolidated Work Summary
**Project:** Bad Photo Detector + Enhancer (PhotoPick)  
**Course:** CIS 5810 | **Group:** 46  
**Prepared for:** November 20, 2026 midterm report  
**Contribution:** Blur and sharpness detection evaluation, dataset preparation, threshold calibration, automated checks, and analysis  

> This is Michael's contribution document for incorporation into the group's combined report. It builds on the team's existing `feature/photopick-iterations` implementation. No teammate algorithm, UI, or earlier iteration document was replaced.

## 1. Summary

The existing PhotoPick system uses the variance of the grayscale Laplacian to detect potentially blurry images. Michael evaluated how this detector behaves across different source images, resolutions, artificially introduced blur types, darkness, and low contrast. He then calibrated an alternative blur threshold using development sources and evaluated it on different sources.

**Result:** On a six-source controlled evaluation cohort, the existing blur cutoff of **75** achieved **76.4% balanced accuracy**. A development-selected candidate cutoff of **approximately 110.895** achieved **78.7%**, with more false positives. At the highest tested resolution, the candidate performed worse. **The production blur threshold remains 75.**

These outcomes describe *synthetic blur-injection detection*; they do not measure human-perceived blur accuracy on unseen consumer photographs.

## 2. Relationship to the team pipeline

The team-provided `photo_quality/core.py` already implements:

- Raw sharpness: variance of the Laplacian on grayscale intensities; `possibly_blurry` if sharpness is below **75**.
- Exposure: mean grayscale intensity, clipping fractions, and related safeguards.
- Contrast: grayscale standard deviation.
- Ranking: **40% smoothed sharpness, 35% exposure, 25% contrast**, with clipping and noise penalties.
- Enhancement: adaptive gamma correction, CLAHE, selective unsharp masking, and user-selectable studio presets.

Michael's work evaluates the existing blur logic through `analyze_image` and does not alter any of these production rules. The team's seven pre-existing iterations, Streamlit interface, tests, requirements, and ranking logic are all preserved.

## 3. Work completed

| Area | Contribution |
| --- | --- |
| Blur and sharpness | Used existing Laplacian variance and blur flag; compared Gaussian and motion blur behavior |
| Dataset preparation | 12 packaged source images; controlled variants at three resolutions |
| Threshold calibration | Development-only balanced-accuracy cutoff selection and a separate source-level evaluation split |
| Evaluation | Confusion matrices, sensitivity, specificity, balanced accuracy, resolution analysis, and failure-case interpretation |
| Code | Reproducible experiment runner and six focused regression checks |
| Documentation | An iteration note that records protocol, numerical results, limitations, and reproducibility |

### Code artifacts

- `scripts/evaluate_blur_threshold_michael.py`: constructs variants, measures existing detector output, calibrates the cutoff on development sources, exports results.
- `tests/test_blur_threshold_michael.py`: adds six focused checks for blur response, current threshold compatibility, case generation, cutoff selection, confusion-matrix reporting, and aspect-ratio handling.
- `docs/iterations/iteration_08_blur_threshold_calibration_michael.md`: records the development iteration and findings.

These were added in [Pull Request #2](https://github.com/lianaxsv/PhotoPicker/pull/2), based on the team's feature branch. That PR does not change existing team files. The earlier standalone prototype PR #1 was closed without merging.

## 4. Experimental design

### 4.1 Dataset and split

We used 12 examples packaged in `scikit-image`, including ordinary photos and scientific or graphical images.

| Cohort | Six source images |
| --- | --- |
| Development | astronaut, coffee, chelsea, camera, rocket, grass |
| Evaluation | coins, brick, gravel, moon, page, hubble_deep_field |

Sources, rather than transformed copies, were split between these two groups. Some sources have appeared in earlier team evaluations, so the evaluation cohort is distinct **within this study**, not a fully unseen cross-project dataset.

Each image was resized with aspect ratio preserved to maximum-side sizes of **128, 256, and 512 pixels**. Five conditions were evaluated:

| Condition | Transformation | Label |
| --- | --- | --- |
| Reference | Unchanged resized image | No injected blur |
| Underexposed control | Pixel intensities multiplied by 0.55 | No injected blur |
| Low-contrast control | Intensity deviation from 127.5 multiplied by 0.55 | No injected blur |
| Gaussian blur | Gaussian sigma = 3.0 pixels | Injected blur |
| Motion blur | Horizontal 15-pixel box kernel | Injected blur |

This creates **12 × 3 × 5 = 180 correlated conditions**: 90 development and 90 evaluation observations, with 72 blur-injected positives and 108 negative controls. The true number of distinct source images is **12**, not 180.

### 4.2 Detection metric and calibration

The existing analyzer measures sharpness through the variance of the Laplacian:

`sharpness = Var(Laplacian(grayscale_image))`

Low variance often indicates less visible edge detail, although naturally low-texture scenes, darkness, and reduced contrast can also produce low values.

The default blur classification is `sharpness < 75`. Candidate thresholds were chosen to maximize **balanced accuracy** on development sources only:

`balanced accuracy = (sensitivity + specificity) / 2`

`sensitivity = TP / (TP + FN)`; `specificity = TN / (TN + FP)`.

Ties were resolved in favor of the threshold closest to the current cutoff. The resulting candidate was **approximately 110.895**. Evaluation-source metrics were not used for selection.

## 5. Quantitative results

The results below were independently reproduced from the stated transformations and cutoff procedure using NumPy, OpenCV, and scikit-image.

### 5.1 Main results

| Cohort (90 cases each) | Threshold | TP | TN | FP | FN | Sensitivity | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 75 | 29 | 52 | 2 | 7 | 80.6% | 96.3% | 88.4% |
| Development | 110.895 | 32 | 52 | 2 | 4 | 88.9% | 96.3% | 92.6% |
| Evaluation | 75 | 25 | 45 | 9 | 11 | 69.4% | 83.3% | 76.4% |
| Evaluation | 110.895 | 28 | 43 | 11 | 8 | 77.8% | 79.6% | 78.7% |

The candidate identified three additional injected-blur conditions on the evaluation cohort (25 to 28 true positives) but increased false positives from 9 to 11. The gain in balanced accuracy was only **2.3 percentage points**.

### 5.2 Resolution sensitivity (evaluation cohort)

| Maximum image side | Existing 75 | Candidate 110.895 |
| --- | ---: | ---: |
| 128 pixels | 73.6% | 77.8% |
| 256 pixels | 77.8% | 86.1% |
| 512 pixels | 77.8% | 72.2% |

This variation indicates that a single global cutoff is sensitive to image resolution.

### 5.3 Blur type and failure cases

- **Gaussian blur:** all **18/18** Gaussian-blurred evaluation conditions were flagged by either threshold.
- **Motion blur:** the existing threshold flagged **7/18**, versus **10/18** for the candidate.
- **Negative controls:** of 54 unchanged, darkened, or low-contrast conditions, **9** were flagged by the current rule and **11** by the candidate.
- **Texture sensitivity:** low-texture images, including the `moon` source, can register as blurry even when no blur operation was injected. That is a synthetic-label discrepancy, not definitive proof that a human would call the image sharp.

## 6. Iteration narrative and decision

**Problem.** Raw Laplacian variance can miss certain injected motion-blur cases and can be artificially low in textureless or dark scenes. A fixed threshold may also respond differently across resolutions.

**Hypothesis.** Calibrating the threshold on multiple controlled blur types and image sources might increase sensitivity without unacceptable false positives.

**Change tested.** We ran a development-only cutoff search. The winning candidate was approximately 110.895, compared with the current production cutoff of 75.

**Evaluation.** The candidate improved balanced accuracy from 76.4% to 78.7% on the separate six-source cohort but increased false positives and reduced balanced accuracy at 512 pixels.

**Decision.** **Keep the production threshold of 75** and treat 110.895 as an experimental candidate. Document the trade-off and gather genuinely new human-labeled photographs before changing the application.

This contribution focuses on *detection*, complementing the team's existing exposure, enhancement, ranking, subjective user-feedback, and deconvolution work.

## 7. Reproducibility and checks

Run from the repository root on Michael's branch (or after PR #2 has been applied):

```bash
python -m pip install -r requirements-dev.txt scikit-image
pytest -q tests/test_blur_threshold_michael.py
python scripts/evaluate_blur_threshold_michael.py --output results/blur_threshold_michael
```

The runner writes `case_metrics.csv` with condition-level results and `summary.md` with cohort and resolution breakdowns to `results/blur_threshold_michael/`. The `results` directory is excluded from Git tracking. The standalone numerical summary was independently checked; teammates should rerun the committed test script before merging.

## 8. Limitations and remaining steps

1. Artificial transformation labels do not equal ground-truth *perceived blur*. The original reference image may itself be soft or intentionally low-texture.
2. Resolutions and transformations are correlated within the same 12 source images; the 180 observations are **not independent photographs**.
3. Although split by source within this experiment, some images were used in prior project work. It is not a generalization result on a newly collected representative camera dataset.
4. Mean brightness, scene detail, motion direction, and spatial resolution affect Laplacian responses. The candidate is not validated for deployment.
5. Next: obtain new permissioned images and human blur ratings, separate calibration from testing by source, compare defocus and real motion blur, and study resolution-aware decision rules while tracking false positives.

## 9. Report-ready paragraph

Michael Ren focused on blur and sharpness evaluation, dataset preparation, and threshold calibration. Using the team's existing Laplacian-variance detector (default threshold 75), he prepared a controlled study with 12 source images, five original or artificially degraded conditions, and three resolutions, for 180 correlated observations. Six sources were used for calibration and six different sources for evaluation. The existing threshold achieved 76.4% balanced accuracy on the evaluation cohort, while a development-selected candidate cutoff near 110.895 achieved 78.7%. The candidate detected more synthetic motion blur but generated additional false positives and performed worse at 512 pixels. Therefore, the production detector was left unchanged. The contribution includes a reproducible evaluation runner, six regression checks, and a documented iteration with failure analysis and next steps. These results describe controlled degradation detection rather than human-rated photographic quality.
