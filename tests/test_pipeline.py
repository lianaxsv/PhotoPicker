"""Basic regression tests; run with python -m unittest discover -s tests."""
import unittest
import numpy as np
import cv2

from photopicker import (adaptive_gamma, calibrate_threshold, detect, gamma_correct,
                         quality_score, rank_photos)


class PipelineTests(unittest.TestCase):
    def test_gamma_brightens(self):
        image = np.full((64, 64, 3), 50, dtype=np.uint8)
        self.assertGreater(float(gamma_correct(image, 0.6).mean()), 50)

    def test_adaptive_gamma_moves_brightness_toward_target(self):
        image = np.full((64, 64, 3), 50, dtype=np.uint8)
        enhanced, gamma = adaptive_gamma(image)
        self.assertLess(abs(float(enhanced.mean()) - 115), 5)
        self.assertLess(gamma, 1)

    def test_threshold_calibration(self):
        cutoff = calibrate_threshold([10, 20, 100, 110], [True, True, False, False])
        self.assertTrue(20 < cutoff < 100)

    def test_blur_flag(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        self.assertLess(cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var(),
                        cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var())

    def test_score_range_and_ranking(self):
        images = [np.full((64, 64, 3), value, dtype=np.uint8)
                  for value in (20, 120, 240)]
        self.assertTrue(all(0 <= quality_score(im) <= 1 for im in images))
        self.assertEqual(rank_photos(images)[0][0], 1)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            detect(np.empty((0, 0, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
