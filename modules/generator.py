import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from modules.crf import pre_defined_conditional_random_field


def load_images_from_folder(folder):
    """
    Loads images from a directory into a list of PIL Image objects.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                images.append(
                    img.copy()
                )  # Use img.copy() if you plan to close the image
        except IOError:
            # You can skip files that aren't images
            print(f"Failed to load {filename}")
    return images


def extract_predictions(images, model):
    """
    This function extracts specific images and predictions based on the provided images and model.

    """

    pred_masks = []

    for image in images:

        # Prediction
        image = np.expand_dims(image, axis=0)
        pred_mask_probs = model.predict(image)
        pred_mask = np.argmax(pred_mask_probs.squeeze(), axis=-1)

        # Collection
        pred_masks.append(pred_mask)

    return pred_masks


def extract_predictions_crf(image_indices, model, pipeline):
    """
    This function extracts specific images, masks, predictions and CRF masks based on the provided indicies, model and dataset.

    """

    image_names = []
    images = []
    masks = []
    pred_masks = []
    crf_masks = []

    for i in image_indices:

        # Extraction
        sample = pipeline.get_sample_by_index(i, 1)
        image = sample[0]
        mask = sample[1]
        mask = tf.squeeze(mask)
        mask = mask.numpy()

        # Prediction
        pred_mask_probs = model.predict(image)
        # image = image.numpy().squeeze()
        pred_mask = np.argmax(pred_mask_probs.squeeze(), axis=-1)
        crf_mask = pre_defined_conditional_random_field(
            image=image.numpy().squeeze(),
            pred_mask_probs=pred_mask_probs,
            inference_iterations=3,
        )

        # Collection
        image_names.append(pipeline.input_img_paths[i])
        images.append(image.numpy().squeeze())
        masks.append(mask)
        pred_masks.append(pred_mask)
        crf_masks.append(crf_mask)

    return image_names, images, masks, pred_masks, crf_masks


def extract_predictions_samples(image_indices, model, pipeline):
    """
    This function extracts specific images, masks and predictions based on the provided indicies, model and dataset.

    """

    image_names = []
    images = []
    masks = []
    pred_masks = []

    for i in image_indices:

        # Extraction
        sample = pipeline.get_sample_by_index(i, 1)
        image = sample[0]
        mask = sample[1]
        mask = tf.squeeze(mask)
        mask = mask.numpy()

        # Prediction
        pred_mask_probs = model.predict(image)
        pred_mask = np.argmax(pred_mask_probs.squeeze(), axis=-1)

        # Collection
        image_names.append(pipeline.input_img_paths[i])
        images.append(image.numpy().squeeze())
        masks.append(mask)
        pred_masks.append(pred_mask)

    return image_names, images, masks, pred_masks
