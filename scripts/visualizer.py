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
from modules.crf import conditional_random_field

# * Model
model_path = "./models/10epoch_32b.keras"
model = keras.models.load_model(model_path)

# * Dataset

BATCH_SIZE = 1
img_dir = "data/img/test"
target_img_dir = "data/masks/test"
pipeline = Pipeline()
dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=img_dir,
    target_img_dir=target_img_dir,
    # max_dataset_len=20,
)

colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]  # White, Red, Green, Blue, Gray
cmap = ListedColormap(colors)


def display_images(titles, images, descriptions, color_map):
    """
    Displays images with their individual titles and array of descriptions.

    Parameters:
    - titles: list of str, titles for each image.
    - images: list of ndarray, image data in a format compatible with plt.imshow.
    - descriptions: list of list of str, each list contains parts of the description for each image.
    """
    if not images or not descriptions or not titles:
        raise ValueError("Titles, images, and descriptions must be non-empty lists.")

    if len(images) != len(descriptions) or len(images) != len(titles):
        raise ValueError(
            "Titles, images, and descriptions must be lists of the same length."
        )

    num_images = len(images)
    plt.figure(figsize=(15, 8))

    for i in range(num_images):
        plt.subplot(2, 2, i + 1)
        # Shortcut
        if images[i].shape[-1] > 10:
            plt.imshow(images[i], cmap=color_map)
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

        # Join the array of details into a single string with line breaks.
        description_text = "\n".join(descriptions[i])

        # Adding the description text below the image
        plt.text(
            0.5,
            -0.1,
            description_text,
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


image_number = 4
# Operation per batch in dataset
for images, masks in dataset.skip(image_number - 1).take(1):
    # Select the first example of the batch
    image = images[0]
    mask = masks[0]

    # Masks
    image_with_batch = np.expand_dims(image, axis=0)

    pred_mask_probs = model.predict(image_with_batch)

    crf_mask = conditional_random_field(
        image=image.numpy(), pred_mask_probs=pred_mask_probs, inference_iterations=5
    )

    # Raw mask
    pred_mask = np.argmax(pred_mask_probs.squeeze(), axis=-1)
    print(pred_mask.shape)
    mask = tf.squeeze(mask)
    mask = mask.numpy()

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
        f"Image\n {pipeline.input_img_paths[image_number]}",
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

    display_images(
        titles=plot_titles,
        images=plot_images,
        descriptions=plot_description,
        color_map=cmap,
    )
