import os
import sys
from datetime import datetime

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras

import modules.model_architectures as model_architectures
from modules.pipeline import Pipeline
from modules.custom_learning_rate import CustomLearningRateScheduler
from modules.loss_functions import multi_class_loss

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
GLOBAL_BATCH_SIZE = 2
EPOCHS = 200

# * Datasets
MAX_NUMBER_SAMPLES = 8

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    batch_size=GLOBAL_BATCH_SIZE,
    per_class_masks=True,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
training_dataset = training_pipeline.dataset

# Validation set
validation_img_dir = "data/img/val"
validation_mask_dir = "data/masks/val"
validation_pipeline = Pipeline()
validation_pipeline.set_dataset_from_directory(
    batch_size=GLOBAL_BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
    per_class_masks=True,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
validation_dataset = validation_pipeline.dataset

strategy = tf.distribute.MirroredStrategy()

# Utilization of multi-GPU acceleration
with strategy.scope():

    # * Model
    model = model_architectures.UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
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
    print("Global batch size:", GLOBAL_BATCH_SIZE)
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

    def lr_schedule(epoch, lr, base_lr, max_epochs):
        """
        Helper function to schedule learning rate

        Inspired by: https://keras.io/guides/writing_your_own_callbacks/#learning-rate-scheduling

        """

        # Polynomial decay - ParseNet approach
        return base_lr * (1 - epoch / max_epochs) ** 0.9

    # * Learning rate parameters

    base_lr = 0.0001  # Target learning rate after warm-up
    initial_lr = 1e-5  # Initial learning rate during warm-up
    warmup_batches = 1  # Number of batches over which to warm up

    milestones = [20, 30, 50]  # Epochs at which to decrease learning rate

    sgd = keras.optimizers.SGD(
        learning_rate=base_lr, weight_decay=0.0005, momentum=0.9, nesterov=True
    )

    # Weights for the Tversky loss function
    # Format: (FP, FN) for each class
    # e.g [Background, Buildings, Trees, Water, Road]
    # Weights decide how much the model should penalize false positives and false negatives
    tversnky_weights = [(2, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    # Penalizes over segmentation of background and under segmentation of buildings

    # Weights for the cross entropy loss function
    # e.g [Background, Buildings, Trees, Water, Road]
    # Weights decide how much the model should penalize each class
    binary_cross_entropy_weights = [0.5, 1, 1, 1, 1]
    # Background and trees are weighted less as they are the dominat classes

    loss = multi_class_loss(tversnky_weights, binary_cross_entropy_weights, DEBUG=True)

    model.compile(
        optimizer=sgd,
        loss=loss,
        metrics=[
            "accuracy",
            # keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        ],
    )

    model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=[
            # model_checkpoint_callback,
            tensorboard,
            early_stopping,
            CustomLearningRateScheduler(
                base_lr, initial_lr, warmup_batches, EPOCHS, lr_schedule, milestones
            ),
        ],
        validation_data=validation_dataset,
        verbose=2,
    )

    print("Training completed")
