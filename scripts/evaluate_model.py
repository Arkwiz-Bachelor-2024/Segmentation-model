import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import keras
import numpy as np
import tensorflow as tf

from modules.metrics import get_class_distribution, get_mIOU, get_OA
from modules.pipeline import Pipeline

# * Components
#TODO change
model = keras.models.load_model("./models/model.keras")
pipeline = Pipeline()
pipeline.set_dataset_from_directory(
    batch_size=1,
    input_img_dir="data/img/test",
    target_img_dir="data/masks/test",
    # max_dataset_len=20,
)
test_dataset = pipeline.dataset

predictions = model.predict(test_dataset)
raw_mask = np.argmax(predictions[1], axis=-1)

# Initialize the MeanIoU metric
miou_metric = keras.metrics.MeanIoU(num_classes=NUM_CLASSES)

# Iterate over the test dataset
for images, true_masks in test_dataset:
    # Predict segmentation masks
    pred_masks_probs = model.predict(images)
    crf_mask = conditional_random_field(
        image=images.numpy(), pred_mask_probs=pred_masks_probs, inference_iterations=5
    )

    # Convert predictions from probabilities to class indices if necessary
    pred_masks = tf.argmax(pred_masks_probs, axis=-1)
    
    # Ensure true_masks and pred_masks are compatible with MeanIoU requirements
    # This might require casting the dtype and reshaping if necessary
    pred_masks = tf.cast(pred_masks, dtype=tf.int32)
    true_masks = tf.cast(true_masks, dtype=tf.int32)
    
    # Update the MeanIoU metric
    miou_metric.update_state(true_masks, pred_masks)

# Retrieve the final mean IoU value
mean_iou = miou_metric.result().numpy()

print(f"Mean IoU over the test set: {mean_iou}")