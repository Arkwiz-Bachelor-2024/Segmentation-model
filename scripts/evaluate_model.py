import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import keras
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from modules.pipeline import Pipeline
from modules.crf import pre_defined_conditional_random_field
from modules.plot import plot_confusion_matrix
from modules.loss_functions import multi_class_tversky_loss

# * Components
weights = [(1.5, 1), (1.5, 1.5), (1, 1), (1.5, 1.5), (1, 1)]
custom_loss_function = multi_class_tversky_loss(weights)
# with tf.keras.utils.custom_object_scope({"loss": custom_loss_function}):
#     model = tf.keras.models.load_model(f"./models/30e_64b_LTV+DA(1)")
# model = keras.models.load_model(f"./models/{os.environ.get('SLURM_JOB_NAME')}")
model = keras.models.load_model(f"./models/U_NET_50e_64b+DA", compile=False)
pipeline = Pipeline()
pipeline.set_dataset_from_directory(
    batch_size=1,
    input_img_dir="data/img/test",
    target_img_dir="data/masks/test",
    max_dataset_len=20,
)
test_dataset = pipeline.dataset
pred_masks_probs = model.predict(test_dataset)
raw_masks = np.argmax(pred_masks_probs, axis=-1)


NUM_CLASSES = 5
# Initialize the MeanIoU metric
raw_miou_metric = keras.metrics.MeanIoU(num_classes=NUM_CLASSES)
crf_miou_metric = keras.metrics.MeanIoU(num_classes=NUM_CLASSES)

for prediction, (image, mask), pred_mask in zip(
    pred_masks_probs, test_dataset, raw_masks
):
    crf_mask = pre_defined_conditional_random_field(
        image=image.numpy().squeeze(),
        pred_mask_probs=prediction,
        inference_iterations=5,
    )

    # Update mIoU
    raw_miou_metric.update_state(mask, pred_mask)
    crf_miou_metric.update_state(mask, crf_mask)

# Extract and flatten masks
masks_only_dataset = test_dataset.map(lambda image, mask: tf.reshape(mask, [-1]))
flattened_masks = []
for flattened_mask in masks_only_dataset:
    flattened_masks.append(flattened_mask.numpy())
masks_flattened = np.concatenate(flattened_masks)
raw_masks_flattened = raw_masks.flatten()

# Compute confusion matrix
cm = confusion_matrix(raw_masks_flattened, masks_flattened, labels=range(NUM_CLASSES))

# Normalize the confusion matrix by row
with np.errstate(divide="ignore", invalid="ignore"):
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[~np.isfinite(cm_normalized)] = 0  # Handle divisions by zero or NaN
cm_percentage = cm_normalized * 100  # Convert to percentages

raw_IoU = raw_miou_metric.result().numpy()
crf_IoU = crf_miou_metric.result().numpy()

cm_classes = ["Background", "Building", "Woodland", "Water", "Road"]
# Check if the directory exists
save_dir = "./docs/models/test/plots/"
if not os.path.exists(save_dir):
    # If the directory does not exist, create it
    os.makedirs(save_dir)

# Now proceed to save your file as before
plot_confusion_matrix(
    cm_percentage,
    cm_classes,
    save_path=f"{save_dir}cm_matrix.png",
)
print(f"Raw mIoU over the test set: {raw_IoU:.2}")
print(f"Crf mIoU over the test set: {crf_IoU:.2}")
