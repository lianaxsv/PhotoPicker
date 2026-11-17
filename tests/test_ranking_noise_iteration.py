"""Regression checks: random sensor noise must not automatically improve photo ranking."""
import cv2
import numpy as np
from photo_quality.core import analyze_image

def checkerboard_with_detail():
    y, x = np.mgrid[:256, :256]
    base = ((x//16 + y//16)%2 * 170 + 35).astype(np.uint8)
    image = cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)
    return image

def test_heavy_noise_does_not_beat_clean_reference():
    original = checkerboard_with_detail()
    rng = np.random.default_rng(5810)
    degraded = np.clip(original.astype(np.float32)+rng.normal(0,28,original.shape),0,255).astype(np.uint8)
    assert analyze_image(original)['quality_score'] >= analyze_image(degraded)['quality_score']

def test_noise_robust_score_is_bounded():
    image = checkerboard_with_detail()
    result = analyze_image(image)
    assert 0 <= result['quality_score'] <= 100
    assert result['sharpness'] >= 0

def test_rank_images_returns_descending_scores():
    from photo_quality.core import rank_images
    clean=checkerboard_with_detail()
    dark=(clean.astype(np.float32)*0.4).astype(np.uint8)
    records=rank_images({'clean':clean,'dark':dark})
    assert records[0]['quality_score'] >= records[1]['quality_score']
