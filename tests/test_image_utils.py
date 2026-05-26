import unittest

import numpy as np
from PIL import Image, ImageDraw

from digit_recognizer.image_utils import (
    format_probabilities,
    normalize_pixels,
    prepare_digit_image,
)


class ImageUtilsTests(unittest.TestCase):
    def test_normalize_pixels_scales_uint8_values(self):
        image = np.array([[0, 127, 255]], dtype=np.uint8)

        normalized = normalize_pixels(image)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_prepare_digit_image_returns_model_ready_shape(self):
        image = Image.new("L", (64, 64), color=255)
        draw = ImageDraw.Draw(image)
        draw.text((24, 16), "7", fill=0)

        prepared = prepare_digit_image(image)

        self.assertEqual(prepared.shape, (1, 28, 28, 1))
        self.assertGreaterEqual(float(prepared.min()), 0.0)
        self.assertLessEqual(float(prepared.max()), 1.0)

    def test_format_probabilities_sorts_descending(self):
        probabilities = np.array([0.01, 0.8, 0.03, 0.02, 0.04, 0.05, 0.01, 0.02, 0.01, 0.01])

        ranked = format_probabilities(probabilities)

        self.assertEqual(ranked[0][0], 1)
        self.assertAlmostEqual(ranked[0][1], 0.8)


if __name__ == "__main__":
    unittest.main()
