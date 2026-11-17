"""Iteration 2 regression tests for intentional two-tone image content.

These reproduce a false positive from Iteration 1. Run with pytest.
"""
import numpy as np
from photo_quality.core import analyze_image


def test_binary_silhouette_is_not_overexposed():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :84] = 255
    result = analyze_image(image)
    assert result["highlight_fraction"] > 0.60
    assert result["shadow_fraction"] > 0.30
    assert "overexposed" not in result["issues"]


def test_binary_silhouette_still_reports_highlight_clipping():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :84] = 255
    assert "highlight_clipping" in analyze_image(image)["issues"]


def test_bright_clipped_scene_with_midtones_remains_detected():
    image = np.full((128, 128, 3), 135, dtype=np.uint8)
    image[:, :55] = 255
    result = analyze_image(image)
    assert result["brightness"] < 205
    assert result["highlight_fraction"] > 0.18
    assert "overexposed" in result["issues"]


def test_mean_brightness_over_205_still_detected():
    image = np.full((128, 128, 3), 220, dtype=np.uint8)
    assert "overexposed" in analyze_image(image)["issues"]
