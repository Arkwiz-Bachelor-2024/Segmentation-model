import os
import sys
import datetime

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from natsort import natsorted

import modules.model_architectures as model_architectures
from modules.pipeline import Pipeline
from modules.loss_functions import multi_class_tversky_loss

"""
This script an initiator to train a segmentation model based upon the specified parameters in the script.

"""
# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available: ", gpus)

# * Hyperparameters
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 3

# * Datasets
MAX_NUMBER_SAMPLES = 300

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    batch_size=BATCH_SIZE,
    max_dataset_len=MAX_NUMBER_SAMPLES,
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

# * Model
model = model_architectures.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

# Loss
# In order of Background, Building, Woodland, Water, Road
# (FP, FN)
weights = [(1, 1), (1,1), (1, 1), (1,1), (1, 1)]
custom_loss_function = multi_class_tversky_loss(weights)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=custom_loss_function)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

CHECKPOINT_FILEPATH = f"./models/{os.environ.get('SLURM_JOB_NAME')}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    mode="min",
    verbose=1,
    save_best_only=True,
)

tensorboard = keras.callbacks.TensorBoard(
    log_dir=f"./docs/logs/{os.environ.get('SLURM_JOB_NAME')}",
    write_steps_per_second=True,
    update_freq="batch",
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

model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback,tensorboard, early_stopping],
    validation_data=validation_dataset,
    verbose=2,
)

print("Training completed")
