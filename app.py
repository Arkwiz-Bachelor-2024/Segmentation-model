#!/bin/sh
# SBATCH --account=share-ie-idi

# ? Makes it so the jobs get executed faster cause of priority.

# from cv2 import applyColorMap
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
import keras
from natsort import natsorted
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# Directory scripts
import models
import pipeline
import visualizer


"""
This script serves as an application for utilizing images and masks to create a semantic segmentation model based upon given specifications. 

"""

# * Hyperparameters

# Model
IMG_SIZE = (512, 512)
# TODO sjekk mengden klasser
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

# Trainig set
training_img_dir = "img/train"
training_mask_dir = "masks/train"
training_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Validation set
validation_img_dir = "img/val"
validation_mask_dir = "masks/val"
validation_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Test set
test_img_dir = "img/test"
test_mask_dir = "masks/test"
test_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=test_img_dir,
    target_img_dir=test_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Creates the model itself
model = models.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy"
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
