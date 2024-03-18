import numpy as np
import tensorflow as tf
from tensorflow import keras
from natsort import natsorted

# Directory scripts
import models
import pipeline

"""
This script serves as an application for utilizing images and masks to create a semantic segmentation model based upon given specifications. 

"""
#* Check if GPU acceleration is available
# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")

# Print the list of available GPUs
print("GPUs Available: ", gpus)

# * Hyperparameters

# Model
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 2


# * Metrics

mean_over_intersection = keras.metrics.MeanIoU(num_classes=NUM_CLASSES)

METRIC = mean_over_intersection


# * Datasets

MAX_NUMBER_SAMPLES = 20

# Trainig set
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Validation set
validation_img_dir = "data/img/val"
validation_mask_dir = "data/masks/val"
validation_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Test set
test_img_dir = "data/img/test"
test_mask_dir = "data/masks/test"
test_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=test_img_dir,
    target_img_dir=test_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)

# Creates the model itself
model = models.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy",
)

# Callback for saving weights
CHECKPOINT_FILEPATH = "./models/model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_best_only=True,
)

print("---------------------------------------------------------------------------------------------------")

print("Data shape")
# Shapes of the data
for x, y in training_dataset.take(1):
    print(x.shape, y.shape)

print("---------------------------------------------------------------------------------------------------")


# Fit the model
model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=model_checkpoint_callback,
    validation_data=validation_dataset,
    verbose=2,
)

print("---------------------------------------------------------------------------------------------------")

print("Predictions: ")

print(predictions.shape)
