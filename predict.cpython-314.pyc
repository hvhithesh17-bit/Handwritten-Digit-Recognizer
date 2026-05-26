from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from digit_recognizer.model import build_cnn_model


def load_mnist():
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (
        x_train[..., np.newaxis],
        y_train,
        x_test[..., np.newaxis],
        y_test,
    )


def plot_history(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train(epochs: int, batch_size: int, model_path: Path, history_path: Path) -> None:
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_cnn_model()
    model.summary()

    callbacks = [
        _make_early_stopping(),
        _make_checkpoint(model_path),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    plot_history(history, history_path)
    print(f"Saved model to {model_path}")
    print(f"Saved training plot to {history_path}")


def _make_early_stopping():
    from tensorflow import keras

    return keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )


def _make_checkpoint(model_path: Path):
    from tensorflow import keras

    model_path.parent.mkdir(parents=True, exist_ok=True)
    return keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        save_best_only=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist_cnn.keras"))
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("reports/training_history.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
        history_path=args.history_path,
    )


if __name__ == "__main__":
    main()

