"""
Module which offers ways to visualize the datset and the predictions made by the model.

"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

from matplotlib.colors import ListedColormap

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from modules.pipeline import get_dataset_from_directory

# * Model

# Path to your saved model
model_path = "./models/10epoch_32b.keras"

# Load the model
model = keras.models.load_model(model_path)


# * Dataset

BATCH_SIZE = 1
img_dir = "data/img/train"
target_img_dir = "data/masks/train"
dataset = get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=img_dir,
    target_img_dir=target_img_dir,
)
# training_dataset_batch = training_dataset[0]
# training_dataset_img_paths = training_dataset[1]
# training_dataset_target_paths = training_dataset[2]

colors = [
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.5, 0.5, 0.5),
]  # White, Red, Green, Blue, Gray
cmap = ListedColormap(colors)

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

image_number = 4000

# Operation per batch in dataset
for images, masks in dataset.skip(image_number - 1).take(20):
    # Select the first example of the batch
    image = images[0]
    mask = masks[0]

    # Masks
    image_with_batch = tf.expand_dims(image, axis=0)
    pred_mask = model.predict(image_with_batch)

    # Assuming `image` is your input image in [0, 255] and `pred_mask_logits` are the logits from your model.
    d = dcrf.DenseCRF2D(
        image.shape[1], image.shape[0], pred_mask.shape[-1]
    )  # Width, Height, NClasses

    pred_mask_probs = np.ascontiguousarray(pred_mask)

    pred_mask_probs = np.transpose(
        pred_mask.squeeze(), (2, 0, 1)
    )  # Now shape [num_classes, height, width]

    pred_mask_probs = np.ascontiguousarray(pred_mask_probs)

    # Now, use this correctly shaped array for unary energies
    U = unary_from_softmax(pred_mask_probs)
    d.setUnaryEnergy(U)

    # Optionally, add pairwise energy terms to enforce smoothness.
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(feats, compat=3)

    feats = create_pairwise_bilateral(
        sdims=(80, 80), schan=(13, 13, 13), img=image.numpy(), chdim=2
    )
    d.addPairwiseEnergy(feats, compat=10)

    # Perform inference to get the refined segmentation.
    Q = d.inference(5)

    # `map_soln` is now your refined segmentation mask.
    crf_mask = np.argmax(Q, axis=0).reshape(image.shape[:2])

    # Raw mask
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]
    mask = tf.squeeze(mask)

    # mIOU metric
    meanIOU_pred = keras.metrics.MeanIoU(num_classes=5)
    meanIOU_pred.update_state(y_true=mask, y_pred=pred_mask)
    meanIOU_pred_score = round(meanIOU_pred.result().numpy(), 2)

    meanIOU_crf = keras.metrics.MeanIoU(num_classes=5)
    meanIOU_crf.update_state(y_true=mask, y_pred=crf_mask)
    meanIOU_crf_score = round(meanIOU_crf.result().numpy(), 2)

    print("Prediction details:")
    print("----------------------------------------------------------------------")
    print("MeanIOU pred_score: ", meanIOU_pred_score)
    print("MeanIOU crf_score: ", meanIOU_crf_score)
    print("----------------------------------------------------------------------")

    # Calculate class distribution percentages for ground truth
    total_pixels = mask.numpy().size
    gt_classes, gt_counts = np.unique(mask.numpy(), return_counts=True)
    gt_distribution_percentages = {
        cls: count / total_pixels for cls, count in zip(gt_classes, gt_counts)
    }

    # Calculate class distribution percentages for predicted mask
    pred_classes, pred_counts = np.unique(pred_mask, return_counts=True)
    pred_distribution_percentages = {
        cls: count / total_pixels for cls, count in zip(pred_classes, pred_counts)
    }

    # Calculate class distribution percentages for crf mask
    crf_classes, crf_counts = np.unique(crf_mask, return_counts=True)
    crf_distribution_percentages = {
        cls: count / total_pixels for cls, count in zip(crf_classes, crf_counts)
    }

    # Calculate accuracy (percentage of correct labels)
    correct_predictions_pred = np.sum(mask.numpy() == pred_mask)
    correct_predictions_crf = np.sum(mask.numpy() == crf_mask)

    pred_accuracy = correct_predictions_pred / total_pixels
    crf_accuracy = correct_predictions_crf / total_pixels
    # Display the ground truth mask

    gt_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in gt_distribution_percentages.items()
        ]
    )

    # Display the predicted mask
    pred_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in pred_distribution_percentages.items()
        ]
    )

    # Display the predicted mask
    crf_distribution_str = ", ".join(
        [
            f"{cls}: {percent:.2%}"
            for cls, percent in crf_distribution_percentages.items()
        ]
    )

    # Plot the image, ground truth, and prediction with distribution percentages
    plt.figure(figsize=(15, 8))

    # Display the original image
    plt.subplot(2, 2, 1)
    plt.title("Image")
    plt.imshow(image)  # Adjust for proper data type if necessary
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title(f"Ground Truth")
    plt.text(
        0.5,
        -0.1,
        f"Distribution: {gt_distribution_str}\n",
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
    )
    plt.imshow(mask.numpy(), cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title(f"Predicted Mask")
    plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.text(
        0.5,
        -0.1,
        f"Overall Accuracy: {pred_accuracy:.2%}\n\nMeanIOU: {meanIOU_pred_score:.2}\n\nPred-Distribution: {pred_distribution_str}",
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
    )
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title(f"CRF")
    plt.imshow(crf_mask, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.text(
        0.5,
        -0.1,
        f"Overall Accuracy: {crf_accuracy:.2%}\n\nMeanIOU: {meanIOU_crf_score:.2}\n\nPred-Distribution: {crf_distribution_str}",
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
