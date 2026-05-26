from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MNIST_IMAGE_SIZE = (28, 28)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Scale uint8-like image pixels to the 0..1 range."""
    array = image.astype("float32")
    if array.max(initial=0) > 1.0:
        array /= 255.0
    return array


def load_digit_image(path: str | Path, invert: bool | None = None) -> np.ndarray:
    """Load an image as a normalized MNIST-shaped tensor.

    Returns an array with shape ``(1, 28, 28, 1)`` suitable for Keras models.
    If ``invert`` is not supplied, the function assumes the digit should be
    bright on a dark background and automatically inverts mostly light images.
    """
    image = Image.open(path)
    return prepare_digit_image(image, invert=invert)


def prepare_digit_image(image: Image.Image, invert: bool | None = None) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.contain(grayscale, MNIST_IMAGE_SIZE)

    canvas = Image.new("L", MNIST_IMAGE_SIZE, color=0)
    left = (MNIST_IMAGE_SIZE[0] - grayscale.width) // 2
    top = (MNIST_IMAGE_SIZE[1] - grayscale.height) // 2
    canvas.paste(grayscale, (left, top))

    array = np.asarray(canvas)
    should_invert = invert if invert is not None else array.mean() > 127
    if should_invert:
        array = 255 - array

    normalized = normalize_pixels(array)
    return normalized.reshape(1, 28, 28, 1)


def format_probabilities(probabilities: np.ndarray) -> list[tuple[int, float]]:
    """Return class probabilities sorted from most to least likely."""
    values = np.asarray(probabilities).reshape(-1)
    if values.size != 10:
        raise ValueError("Expected exactly 10 class probabilities.")
    return sorted(
        ((digit, float(probability)) for digit, probability in enumerate(values)),
        key=lambda item: item[1],
        reverse=True,
    )

