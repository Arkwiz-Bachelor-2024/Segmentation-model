import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import matplotlib.cm as cm

# TODO refactor to new location
from modules.generator import extract_predictions
from modules.metrics import get_class_distribution

# * Pre-determined Colourmap
# White, Red, Green, Blue, Gray in RGB
colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]
cmap = ListedColormap(colors)


def generate_masks_with_details(images, model_name):
    """
    Given an array of images and the name of the chosen model, this function will generate masks and class distribution details for each image.

    Returns a dictonary containing the respective masks and their associated class distribution.

    """
    if images is None:
        raise ValueError("No images provided")

    if model_name is None:
        raise ValueError("No model name provided")

    try:
        # Load and predict images
        model_path = f"./models/{model_name}"
        model = keras.models.load_model(model_path)
        masks = extract_predictions(images, model)

    except Exception as e:
        print(f"An error occured while generating masks: {e}")

    results = []

    for i in range(len(masks)):

        class_distribution = get_class_distribution(masks[i])

        # Convert mask labels to image
        mapped_data = cm(masks[i] / masks.max())  # Normalize the data
        image_data = (mapped_data * 255).astype(np.uint8)

        # Create the image
        mask_img = Image.fromarray(image_data, "RGBA")

        results.append(
            {
                f"Mask number.{i + 1}": {
                    "Mask image": mask_img,
                    "Class Distribution": class_distribution,
                }
            }
        )

    return results
