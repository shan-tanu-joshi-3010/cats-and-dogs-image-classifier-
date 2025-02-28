# cats-and-dogs-image-classifier
This project is a deep learning-based classifier that distinguishes between images of cats and dogs using Convolutional Neural Networks (CNNs). The model is trained on labeled datasets and evaluated for accuracy.

## Features
- Image classification for cats and dogs
- Uses CNNs for feature extraction
- Supports pretrained models for fine-tuning
- Provides real-time prediction

## Code Samples
try:
  # This command only in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# Get project files
!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

!unzip cats_and_dogs.zip

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

## Installation & Usage
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/cats-and-dogs-image-classifier.git
   ```
2. Install dependencies:
   bash
   pip install -r requirements.txt
   ```
3. Train the model:
   bash
   python train.py
   ```
4. Predict an image:
   bash
   python predict.py --image path/to/image.jpg
