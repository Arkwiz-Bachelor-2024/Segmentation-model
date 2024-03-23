import keras
import numpy as np


def get_mIOU(ground_truth, predictions, classes):
    # mIOU metric
    mIOU = keras.metrics.MeanIoU(num_classes=classes)
    mIOU.update_state(y_true=ground_truth, y_pred=predictions)
    mIOU_score = round(mIOU.result().numpy(), 2)

    return mIOU_score


def get_class_distribution(image):

    # Calculate class distribution percentages for ground truth
    total_pixels = image.size

    classes, counts = np.unique(image, return_counts=True)
    distribution_percentages = {
        cls: count / total_pixels for cls, count in zip(classes, counts)
    }

    return distribution_percentages


def get_OA(ground_truth, predictions):

    true_positives = np.sum(ground_truth == predictions)

    overall_accuracy = true_positives / ground_truth.size

    return overall_accuracy
