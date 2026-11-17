"""Regression: high highlight clipping must count as probable overexposure."""
import numpy as np
from photo_quality.core import analyze_image


def test_clipped_highlights_trigger_overexposure_even_with_moderate_mean():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    image[:, :30] = 255
    result = analyze_image(image)
    assert 100 < result['brightness'] < 205
    assert result['highlight_fraction'] >= .29
    assert 'overexposed' in result['issues']


def test_unclipped_gradient_not_overexposed():
    grad = np.tile(np.linspace(40, 210, 256, dtype=np.uint8), (256, 1))
    image = np.stack((grad,) * 3, axis=-1)
    assert 'overexposed' not in analyze_image(image)['issues']
