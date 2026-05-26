# Handwritten Digit Recognizer

Train a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset.

This project includes:

- A TensorFlow/Keras CNN training script
- Image preprocessing for custom handwritten digit images
- A command-line predictor
- A polished Streamlit website with a manual draw-and-train studio
- Lightweight tests for preprocessing utilities

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── src/
│   └── digit_recognizer/
│       ├── __init__.py
│       ├── image_utils.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
└── tests/
    └── test_image_utils.py
```

## Setup

TensorFlow wheels are not always available for the newest Python versions. If `pip install tensorflow` fails on Python 3.14, create an environment with Python 3.11 or 3.12 and install there.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train the CNN

```powershell
python -m digit_recognizer.train --epochs 5 --model-path models/mnist_cnn.keras
```

The script downloads MNIST automatically through Keras, trains the CNN, evaluates it on the test set, and saves:

- `models/mnist_cnn.keras`
- `reports/training_history.png`

On a typical machine, 5 epochs should reach about 99% test accuracy.

## Predict a Custom Image

Use a dark digit on a light background or a light digit on a dark background. The preprocessing step converts the image to the 28x28 MNIST format.

```powershell
python -m digit_recognizer.predict path\to\digit.png --model-path models/mnist_cnn.keras
```

## Run the Demo App

```powershell
streamlit run app.py
```

Use the manual studio to draw digits, label them, add training samples, train the browser model, and predict directly from your drawing. After training the TensorFlow CNN, you can also upload an image and the app will show the predicted class with class probabilities.

## Run Tests

```powershell
python -m unittest discover -s tests
```
