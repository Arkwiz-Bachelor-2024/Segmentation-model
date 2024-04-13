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
from modules.loss_functions import multi_class_tversky_loss
from modules.custom_learning_rate import CustomLearningRateScheduler

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
BATCH_SIZE = 12
EPOCHS = 100

# * Datasets
MAX_NUMBER_SAMPLES = 2

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    batch_size=BATCH_SIZE,
    # per_class_masks=True,
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
    # per_class_masks=True
    # max_dataset_len=MAX_NUMBER_SAMPLES
)
validation_dataset = validation_pipeline.dataset

strategy = tf.distribute.MirroredStrategy()

# Utilization of multi-GPU acceleration
with strategy.scope():

    # * Model
    model = model_architectures.DeeplabV3Plus(
        img_size=IMG_SIZE, num_classes=NUM_CLASSES
    )

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=15,
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
    print("Total batch size:", BATCH_SIZE)
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


    LR_SCHEDULE = [
        # (epoch to start, learning rate) tuples
        (1, 0.01),
        (80, 0.001),
        (120, 0.0001),
        (160, 0.00001),
    ]

    def lr_schedule(epoch, lr):
        """
        Helper function to retrieve the scheduled learning rate based on epoch.

        Inspired by: https://keras.io/guides/writing_your_own_callbacks/#learning-rate-scheduling 
        
        """
        # Update from schedule
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]

        # Other epochs
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
                return lr

        return lr

    #* Learning rate parameters

    initial_lr = 0.0001   # Initial learning rate for warm-up
    peak_lr = 0.01        # Target learning rate after warm-up
    warmup_batches = 20   # Number of batches over which to warm up
    post_warmup_lr = 0.01 # Learning rate after warm-up, before switch
    switch_epoch = 40     # Epoch to switch from Adam to SGD
    post_switch_lr = 0.001

    adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.5, weight_decay=1e-4)

    model.compile(
        optimizer=adam,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        ],
    )

    model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            tensorboard,
            early_stopping,
            CustomLearningRateScheduler(initial_lr,peak_lr, warmup_batches, post_warmup_lr, switch_epoch, post_switch_lr, schedule=lr_schedule),
        ],
        validation_data=validation_dataset,
        verbose=2,
    )

    print("Training completed")
