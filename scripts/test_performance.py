"""
Module which offers ways to visualize the datset and the predictions made by the model.

"""

import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import data as tf_data
from tensorflow import io as tf_io
from tensorflow import image as tf_image
from matplotlib.colors import ListedColormap
from modules.plot import simple_image_display
from modules.generator import load_images_from_folder, extract_predictions

# * Components
model = keras.models.load_model(
    "./models/Deeplabv3Plus_100e_4b_Centropy_adaptive_sgd+DA+DO", compile=False
)

# * Customization
colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]  # White, Red, Green, Blue, Gray
cmap = ListedColormap(colors)

images = load_images_from_folder(folder="data/test")

for i in range(len(images)):
    images[i] = tf_image.resize(images[i], (512, 512))
    images[i] = tf.cast(images[i], tf.float32) / 255.0
    images[i] = tf_image.convert_image_dtype(images[i], "float32")

# input_img = tf_image.resize(images[0], (512, 512))
# input_img = tf.cast(input_img, tf.float32) / 255.0
# # Dtype describes how the bytes of the image are to be interpeted, e.g the format of the image.
# input_img = tf_image.convert_image_dtype(input_img, "float32")


pred_masks = extract_predictions(images=images[2:-2], model=model)

pred_images = []
titles = []
index = 0
for image in pred_masks:
    titles.append(str(index))
    pred_images.append(image)
    index += 1


simple_image_display(
    images=pred_images[0:8],
    color_map=cmap,
    titles=titles[0:8],
    descriptions=None,
)
