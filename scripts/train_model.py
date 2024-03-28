import os
import sys

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from natsort import natsorted

import modules.model_architectures as model_architectures
from modules.crf import conditional_random_field
from modules.pipeline import Pipeline

"""
This script an initiator to train a segmentation model based upon the specified parameters in the script.

"""
# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available: ", gpus)

# * Hyperparameters
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 10

# * Datasets
MAX_NUMBER_SAMPLES = 20

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    batch_size=BATCH_SIZE,
    # max_dataset_len=MAX_NUMBER_SAMPLES,
)
training_dataset = training_pipeline.dataset

# Validation set
validation_img_dir = "data/img/val"
validation_mask_dir = "data/masks/val"
validation_pipeline = Pipeline()
validation_pipeline.set_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
)
validation_dataset = validation_pipeline.dataset
# Test set
test_img_dir = "data/img/test"
test_mask_dir = "data/masks/test"
test_pipeline = Pipeline()
test_pipeline.set_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=test_img_dir,
    target_img_dir=test_mask_dir,
)
test_dataset = test_pipeline.dataset

# Creates the model itself
model = model_architectures.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
)

# Callback for saving model
CHECKPOINT_FILEPATH = f"./models/{os.environ.get('SLURM_JOB_NAME')}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    mode="auto",
    verbose=1,
    save_best_only=True,
)

print(
    "---------------------------------------------------------------------------------------------------"
)
print("Data shape")
# Shapes of the data
for x, y in training_dataset.take(1):
    print(f"Image shape: {x.shape}")
    print(f"Mask shape: {y.shape}")

print("Training dataset: ", training_dataset)
print(
    "---------------------------------------------------------------------------------------------------"
)

# Train
model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=model_checkpoint_callback,
    validation_data=validation_dataset,
    verbose=2,
)
