# from cv2 import applyColorMap
import numpy as np
from tensorflow import keras
import keras
import os
import sys
from natsort import natsorted


# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Directory scripts
import modules.model_architectures as model_architectures
from modules.pipeline import Pipeline

"""
This script serves as an application for utilizing images and masks to create a semantic segmentation model based upon given specifications. 

"""

# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available: ", gpus)

# * Hyperparameters

IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 10
BATCH_SIZE = 8
EPOCHS = 2

# * Datasets

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    training_img_dir=training_img_dir,
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
test_pipeline.get_dataset_from_directory(
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

# Callback for saving weights
CHECKPOINT_FILEPATH = "./models/model.keras"
#TODO add model name to enviroment upon calling job
CHECKPOINT_FILEPATH = f"./models/{os.environ.get("MODEL_NAME")}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    mode="max",
    verbose=1,
    save_best_only=True,
)

print("---------------------------------------------------------------------------------------------------")

print("Data shape")

# Shapes of the data
for x, y in training_dataset.take(1):
    print(f"Image shape: {x.shape}")
    print(f"Mask shape: {y.shape}")

print("---------------------------------------------------------------------------------------------------")


# Fit the model
model.fit(
    training_dataset,
    epochs=EPOCHS,
    callbacks=model_checkpoint_callback,
    validation_data=validation_dataset,
    verbose=2,
)

predictions = model.predict(test_dataset)

print(predictions)

mask = np.argmax(predictions[1], axis=-1)

# TODO Predict whole test set on mIOU metric and make a custom output
