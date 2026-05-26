from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from digit_recognizer.image_utils import format_probabilities, load_digit_image


def predict_digit(image_path: Path, model_path: Path) -> tuple[int, list[tuple[int, float]]]:
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    image = load_digit_image(image_path)
    probabilities = model.predict(image, verbose=0)[0]
    ranked = format_probabilities(probabilities)
    return int(np.argmax(probabilities)), ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a digit from an image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist_cnn.keras"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digit, ranked = predict_digit(args.image_path, args.model_path)
    print(f"Predicted digit: {digit}")
    print("Probabilities:")
    for label, probability in ranked:
        print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()

