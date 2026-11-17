# Iteration 6 — Human blur-preference evaluation (in progress)

**Status: awaiting human ratings. No production fix deployed.**

## Motivation

A real user reported that Balanced and Vivid visually worsened a motion-blurred personal photo, despite the heuristic ranking score increasing. This is one subjective observation, not an independent benchmark. The personal image has **not** been committed to GitHub.

## External dataset identified

[Aleksey Alekseev's Blur Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset), [repository documentation](https://github.com/Kwentar/blur_dataset), CC0 public-domain per the dataset's published metadata. It contains 350 sharp / motion blur / defocus triplets (1,050 images). Different captures of the same scene are **not pixel aligned**; do not compute PSNR/MAE using its sharp images as exact references.

Because external image downloads are blocked in the current processing runtime, this dataset has been **identified but not downloaded or evaluated here**. Script `scripts/make_kwentar_blur_review.py` prepares authentic dataset comparisons when files are available locally. This is not a claim of online dataset results.

## Preliminary rating sheet for immediate user feedback

Eight existing *genuine source photographs* bundled with scikit-image / Matplotlib were subjected to **artificial blur** (linear motion kernels or Gaussian defocus), and processed using the current `enhance_image` Auto pipeline and `enhance_studio(..., intensity=.8, style="Balanced")`. Images and metrics are available in the conversation artifact `blur_rater_package.zip`, not pushed to the repository. Subjects: coffee, Chelsea cat, astronaut, rocket, camera photographer, Grace Hopper portrait, grass texture, Hubble Deep Field.

Every photo was measured for blur flags, raw Laplacian variance, the applied automatic filter operations, and mean pixel change from input. No participant ratings have been collected yet.

**These eight images are NOT samples from the Kwentar internet dataset**. They are controlled degradations of photographs bundled with Python scientific packages; source photos may have different licensing. They must not be represented as independently collected real motion-blurred camera photos.

## Rating protocol

For each numbered image, rate which output looks best **as a photograph**, not which has highest contrast: blurry input, Auto, or Balanced 80%, or choose Tie. Optionally comment about halos, noise, oversharpening, color and lost detail. Record preference **before changing code**, preserving both preferred and not preferred cases.

## Next decisions

1. Collect real user judgments for the initial 8 examples.
2. Obtain actual CC0 dataset locally and prepare 8+ real motion-blur / defocus comparisons with the provided script; use different source subjects from calibration.
3. Create a no-op/safeguard candidate for severe blur; measure **preferred-output rate** vs current Auto/Balanced, and false negatives on milder blur.
4. Only promote an algorithm fix if it improves on the independent human rating set. Keep separate test, fix, evaluation and documentation commits.
