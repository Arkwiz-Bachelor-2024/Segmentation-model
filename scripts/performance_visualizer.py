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
from modules.crf import pre_defined_conditional_random_field
from modules.plot import simple_image_display
from modules.generator import extract_predictions_crf

# * Components
model = keras.models.load_model(
    "./models/Deeplabv3Plus_50e_32b_Centropy_SGD_warmup+DO_mild+DA_mild",
    compile=False,
)
pipeline = Pipeline()
pipeline.set_dataset_from_directory(
    batch_size=1,
    input_img_dir="data/img/test",
    target_img_dir="data/masks/test",
    # max_dataset_len=20,
)
dataset = pipeline.dataset

# * Customization
colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]  # White, Red, Green, Blue, Gray
cmap = ListedColormap(colors)

# * Extraction
# [ 537 1014 1190   71   84 1305 1215   86 1184  547]
image_indices = [537, 1014, 1190, 71, 84, 1305, 1215, 86, 1184, 547]
# images__
image_names, images, masks, pred_masks, crf_masks = extract_predictions_crf(
    image_indices, model, pipeline
)

# * Visualization
for image_name, image, mask, pred_mask, crf_mask in zip(
    image_names, images, masks, pred_masks, crf_masks
):

    print("----------------------------------------------------------------------")
    print("Image details(shape, type):")
    print("Image", image.shape, type(image))
    print("Mask", mask.shape, type(mask))
    print("Pred_mask", pred_mask.shape, type(pred_mask))
    print("CRF_mask", crf_mask.shape, type(crf_mask))
    print("----------------------------------------------------------------------")

    meanIOU_crf_score = get_mIOU(mask, crf_mask, mask.shape[-1])
    meanIOU_pred_score = get_mIOU(mask, pred_mask, mask.shape[-1])

    OA_pred = get_OA(mask, pred_mask)
    OA_crf = get_OA(mask, crf_mask)

    gt_distribution_percentages = get_class_distribution(image=mask)
    pred_distribution_percentages = get_class_distribution(image=pred_mask)
    crf_distribution_percentages = get_class_distribution(image=crf_mask)

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
    crf_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in crf_distribution_percentages.items()
        ]
    )

    print("Prediction details:")
    print("----------------------------------------------------------------------")
    print("MeanIOU pred_score: ", meanIOU_pred_score)
    print("MeanIOU crf_score: ", meanIOU_crf_score)
    print("----------------------------------------------------------------------")

    plot_images = [image, mask, pred_mask, crf_mask]
    plot_titles = [
        f"Image\n {image_name}",
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
    plot_crf_mask_details = [
        f"Class Distribution: {crf_distribution_str}",
        f"mIOU: {meanIOU_crf_score:.2} ",
        f"OA: {OA_crf:.2}",
    ]

    plot_description = [
        plot_image_details,
        plot_mask_details,
        plot_pred_mask_details,
        plot_crf_mask_details,
    ]

    simple_image_display(
        titles=plot_titles,
        images=plot_images,
        descriptions=plot_description,
        color_map=cmap,
    )
