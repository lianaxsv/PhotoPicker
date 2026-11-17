# PhotoPick — Bad Photo Detector + Enhancer

CIS 5810, Group 46. PhotoPick ranks image quality, identifies common defects, and applies adjustable enhancement filters.

## Features

- Three-column image gallery with selectable photos and quality scores.
- Detection for likely blur (variance of Laplacian), exposure (brightness and clipping), and contrast (standard deviation).
- Side-by-side, reveal-slider, and difference-map before/after comparisons.
- Three enhancement styles: Natural, Balanced, and Vivid, with strength control.
- Adaptive gamma correction, CLAHE on LAB luminance, and selective unsharp masking.
- Batch downloads and ranking CSV export.
- CLI image processing and tests.

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The demo gallery displays **synthetic examples** of common image defects, so the app opens with content even without uploads. Choose **Upload photos** in the sidebar to analyze real photos.

## Run tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Batch CLI

```bash
python cli.py path/to/photos path/to/results --mode auto
```

## Evaluation

```bash
python evaluate.py /path/to/reference.jpg /path/to/output.csv
```

The evaluation script creates controlled underexposures and compares fixed and adaptive gamma using MAE and PSNR. It is not a substitute for evaluation on a broad real-world dataset.

The quality ranking is **heuristic**, not a learned aesthetic rating: 40% sharpness, 35% exposure, 25% contrast, less a clipping penalty. Blur thresholds can depend on content and image dimensions.

The project's original proposal includes measured gamma correction values on a controlled illustration. Later example journal results and AI-generated visual demonstrations should **not** be interpreted as measured benchmarks without reproducing the experiments. See `docs/README.md`.

## Structure

- `app.py` — Streamlit gallery and enhancement studio
- `photo_quality/core.py` — detection, ranking, filtering
- `cli.py` — batch processing
- `evaluate.py` — reproducible reference-based experiment
- `tests/` — automated tests
- `docs/README.md` — methodological and iteration notes

## Next steps

Calibrate thresholds using labeled real photographs, evaluate against held-out references, compare rankings to human preference, test failure cases, and validate the Streamlit interface in a browser.
