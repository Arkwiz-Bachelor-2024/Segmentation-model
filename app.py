# from cv2 import applyColorMap
import numpy as np
import tensorflow as tf
import keras
from natsort import natsorted

# Directory scripts
import models
import pipeline


"""
This script serves as an application for utilizing images and masks to create a semantic segmentation model based upon given specifications. 

"""

# * Hyperparameters

# Model
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 1

# * Metrics
mean_over_intersection = keras.metrics.MeanIoU(
    NUM_CLASSES,
    name=None,
    dtype=None,
    ignore_class=None,
    sparse_y_true=True,
    sparse_y_pred=True,
)

METRIC = mean_over_intersection


# * Datasets

MAX_NUMBER_SAMPLES = 20

# Training set
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
training_dataset_batch = training_dataset[0]
training_dataset_img_paths = training_dataset[2]
training_dataset_target_paths = training_dataset[2]

# Validation set
validation_img_dir = "data/img/val"
validation_mask_dir = "data/masks/val"
validation_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
validation_dataset_batch = validation_dataset[0]
validation_dataset_img_paths = validation_dataset[1]
validation_dataset_target_paths = validation_dataset[2]


# Test set
test_img_dir = "data/img/test"
test_mask_dir = "data/masks/test"
test_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=test_img_dir,
    target_img_dir=test_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
test_dataset_batch = training_dataset[0]
test_dataset_img_paths = training_dataset[1]
test_dataset_target_paths = training_dataset[2]


# Test set

# Creates the model itself
model = models.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=METRIC,
)

# Callback for saving weights
CHECKPOINT_FILEPATH = "./ckpt/checkpoint.model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    monitor=METRIC,
    mode="max",
    verbose=1,
    save_best_only=True,
)

# Fit the model
model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=model_checkpoint_callback,
    validation_data=validation_dataset,
    verbose=1,
)

predictions = model.predict(training_dataset)

print(predictions)
