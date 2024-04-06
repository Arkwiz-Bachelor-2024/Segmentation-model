import os
import sys
from datetime import datetime

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import keras

import numpy as np
import tensorflow as tf
from tensorflow import keras

import modules.model_architectures as model_architectures
from modules.pipeline import Pipeline
from modules.loss_functions import multi_class_tversky_loss

"""
This script an initiator to train a segmentation model based upon the specified parameters in the script.

"""

# * Enviroment
print(
    "---------------------------------------------------------------------------------------------------"
)
print("Enviroment:")

print("TensorFlow version:", tf.__version__)

# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available: ", gpus)

print(
    "---------------------------------------------------------------------------------------------------"
)

# * Hyperparameters
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 200

# * Datasets
# MAX_NUMBER_SAMPLES = 50

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    batch_size=BATCH_SIZE,
    per_class_masks=True,
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
    per_class_masks=True
    # max_dataset_len=MAX_NUMBER_SAMPLES
)
validation_dataset = validation_pipeline.dataset

strategy = tf.distribute.MirroredStrategy()

# Utilization of multi-GPU acceleration
with strategy.scope():

    # * Model
    model = model_architectures.UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    # # In order of Background, Building, Woodland, Water, Road
    # # (FP, FN)
    weights = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    custom_loss_function = multi_class_tversky_loss(weights)


    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="min",
    )

    CHECKPOINT_FILEPATH = f"./models/{os.environ.get('SLURM_JOB_NAME')}"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    mode="min",
    verbose=1,
    save_best_only=True,
    )

    tensorboard = keras.callbacks.TensorBoard(
    log_dir=f"./docs/models/{os.environ.get('SLURM_JOB_NAME')}/logs/{datetime.now().strftime('%d.%m.%Y-%H_%M')}",
    )

    # * Logging
    print("Details:")
    print(
    "---------------------------------------------------------------------------------------------------"
    )
    print("Classes: ", NUM_CLASSES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs", EPOCHS)
    print(
    "---------------------------------------------------------------------------------------------------"
    )
    print("Data shape")
    # Shapes of the data
    for x, y in training_dataset.take(1):

        print(f"Image shape: {x.shape}")
        print(f"Mask shape: {y.shape}")

    print("Training dataset: ", training_dataset)
    print("Validation dataset:", validation_dataset)
    print(
    "---------------------------------------------------------------------------------------------------"
    )

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=custom_loss_function)

    model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, tensorboard, early_stopping],
        validation_data=validation_dataset,
        verbose=2,
        run_eagerly=True
    )

    print("Training completed")
