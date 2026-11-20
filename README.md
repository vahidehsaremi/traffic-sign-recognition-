markdown
# Traffic Sign Recognition

A Python command-line application and computer vision pipeline for recognizing and classifying traffic signs using deep learning and Keras. Built as part of VAYDA SYSTEM infrastructure.

## Requirements

- Python 3.10 or 3.11
- A trained Keras model saved at `models/traffic_sign_model.h5`
- The model must accept RGB images resized to 32x32 pixels

## Installation

pip install -r requirements.txt

## Usage

To run prediction on a sample image:
python main.py --image path/to/sign.jpg

## Features
- Real-time image preprocessing and normalization
- Robust classification pipeline for standard traffic categories
- Evaluation scripts and metric tracking
```