"""Regression checks for Michael's evaluation-only blur calibration."""
import numpy as np
import pytest
from photo_quality.core import analyze_image, DEFAULT_THRESHOLDS
from scripts.evaluate_blur_threshold_michael import (
    balanced_accuracy, choose_threshold, confusion_stats,
    degraded_variants, motion_blur, scaled_photo,
)

def test_existing_production_threshold_is_unchanged():
    img = np.random.default_rng(5810).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    result = analyze_image(img)
    assert DEFAULT_THRESHOLDS['blur'] == 75.0
    assert ('possibly_blurry' in result['issues']) == (result['sharpness'] < 75.0)

def test_motion_blur_reduces_sharpness():
    img = np.random.default_rng(46).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert analyze_image(motion_blur(img))['sharpness'] < analyze_image(img)['sharpness']

def test_five_controlled_variants():
    img = np.random.default_rng(6).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    variants = degraded_variants(img)
    assert len(variants) == 5
    assert all(a.shape == img.shape and a.dtype == np.uint8 for a in variants.values())

def test_development_threshold_separates_toy_data():
    threshold, acc = choose_threshold([3, 8, 70, 95], [True, True, False, False], 75.0)
    assert 8 < threshold <= 70
    assert acc == 1.0

def test_single_class_error():
    with pytest.raises(ValueError):
        balanced_accuracy([True, True], [True, False])

def test_confusion_and_aspect_ratio():
    rows = [
        {'applied_blur': 1, 'pred': 1}, {'applied_blur': 1, 'pred': 0},
        {'applied_blur': 0, 'pred': 0}, {'applied_blur': 0, 'pred': 1},
    ]
    m = confusion_stats(rows, 'pred')
    assert (m['tp'], m['tn'], m['fp'], m['fn']) == (1, 1, 1, 1)
    assert m['balanced_accuracy'] == 0.5
    assert scaled_photo(np.zeros((60, 120, 3), dtype=np.uint8), 256).shape[:2] == (128, 256)
