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

from modules.metrics import get_class_distribution
from modules.metrics import get_mIOU
from modules.metrics import get_OA
from matplotlib.colors import ListedColormap
from modules.pipeline import Pipeline
from modules.plot import simple_image_display
from modules.generator import load_images_from_folder, extract_predictions
from utils.json_masks import process_json_files

# * Customization
colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]  # White, Red, Green, Blue, Gray
cmap = ListedColormap(colors)

# * Components
model = keras.models.load_model(
    "./models/UNETplus_2x_100e_16b_Poly_Adam_low_wBCE_milestones_warmup+DA_mid",
    compile=False,
)

pipeline = Pipeline()
pipeline.set_dataset_from_directory(
    batch_size=1,
    input_img_dir="data/test/img",
    target_img_dir="data/test/img",
    # max_dataset_len=20,
)
dataset = pipeline.dataset
images = []

for image, mask in dataset:
    images.append(image.numpy().squeeze())

masks = process_json_files("data/test/mask_data")
pred_masks = extract_predictions(images, model)

# * Visualization
for image, mask, pred_mask in zip(images, masks, pred_masks):

    print("----------------------------------------------------------------------")
    print("Image details(shape, type):")
    print("Image", image.shape, type(image))
    print("Mask", mask.shape, type(mask))
    print("Pred_mask", pred_mask.shape, type(pred_mask))
    print("----------------------------------------------------------------------")

    meanIOU_pred_score = get_mIOU(mask, pred_mask, mask.shape[-1])

    OA_pred = get_OA(mask, pred_mask)

    gt_distribution_percentages = get_class_distribution(image=mask)
    pred_distribution_percentages = get_class_distribution(image=pred_mask)

    gt_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in gt_distribution_percentages.items()
        ]
    )
    pred_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in pred_distribution_percentages.items()
        ]
    )

    print("Prediction details:")
    print("----------------------------------------------------------------------")
    print("MeanIOU pred_score: ", meanIOU_pred_score)
    print("----------------------------------------------------------------------")

    plot_images = [image, mask, pred_mask]
    plot_titles = [
        f"Test Image",
        "Ground truth",
        "Prediction mask",
        "CRF mask",
    ]

    plot_image_details = []
    plot_mask_details = [f"Class Distribution: {gt_distribution_str}"]
    plot_pred_mask_details = [
        f"Class Distribution: {pred_distribution_str}",
        f"mIOU: {meanIOU_pred_score:.2} ",
        f"OA: {OA_pred:.2}",
    ]

    plot_description = [
        plot_image_details,
        plot_mask_details,
        plot_pred_mask_details,
    ]

    simple_image_display(
        titles=plot_titles,
        images=plot_images,
        descriptions=plot_description,
        color_map=cmap,
    )
