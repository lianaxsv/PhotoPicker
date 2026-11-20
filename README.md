# PhotoPicker — Bad Photo Detector + Enhancer

CIS 5810, Group 46. Grace Ye, Liana Veerasamy, Michael Ren.

This repository contains an **experimental computer-vision pipeline** for flagging blur, exposure, and contrast problems, applying basic enhancements, and ranking related photographs.

## Midterm iterations (November 20, 2026)

The original proposal documented a fixed gamma=0.8 exposure baseline (darkened illustration MAE 61.64 before correction, 42.98 afterward) and a gamma=0.6 iteration (MAE 16.52). Those figures come from the **earlier proposal**, not the experiments in this repository.

- **Iteration 2 — detection calibration:** tune Laplacian-variance blur threshold using 12 synthetic training image seeds, then evaluate on 8 disjoint seeds (24 held-out examples). The CSV records labels and classifications.
- **Iteration 3 — adaptive enhancement:** compare no correction, fixed gamma=0.8, fixed gamma=0.6, adaptive gamma and CLAHE on 24 darkened, held-out synthetic images. Report MAE and PSNR relative to the undarkened reference.
- **Iteration 4 — weighted ranking:** assess a 45% sharpness, 35% exposure, 20% contrast score on eight synthetic sets of the same scene with blur, darkness, and contrast perturbations. Check whether the unmodified reference ranks first.

**Caution:** these are controlled, synthetic tests. No real-photograph detection accuracy or agreement with human rankings is implied. Sharpness thresholds depend on resolution and content. The scoring weights are provisional. CLAHE and gamma can sometimes make an image look worse.

## Reproduce

Requires Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python experiments.py --output results
```

Read `results/summary.txt` and the three CSVs to obtain the **actual measured results**; do not copy unrun numbers into the midterm report. The experiment also saves a set of illustrative PNGs. Results are generated locally and ignored by Git.

## Implementation

- `photopicker.py`: image-quality metrics, threshold selection, issue flags, fixed and adaptive gamma, conditional CLAHE, heuristic quality ranking
- `experiments.py`: reproducible synthetic dataset and three evaluation iterations
- `tests/test_pipeline.py`: basic automated regression tests

## Next steps

Evaluate on real photographs with human-labeled blur and exposure issues; calibrate thresholds by resolution, scene type, and lighting. Compare quality rankings with human preferences. Add a Streamlit or Gradio interface and before-and-after display. Assign final team integration according to the existing project proposal.

## Contributors

- Grace Ye — photo ranking, integration, interface, evaluation
- Liana Veerasamy — exposure, contrast, enhancement, evaluation
- Michael Ren — blur and sharpness, image testing, dataset preparation, threshold calibration
