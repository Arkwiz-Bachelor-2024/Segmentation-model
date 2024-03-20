# from cv2 import applyColorMap
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
# * Check if GPU acceleration is available
# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")

# Print the list of available GPUs
print("GPUs Available: ", gpus)

# * Hyperparameters

# Model
IMG_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 10

# * Datasets

# Trainig set
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
)
# training_dataset_batch = training_dataset[0]
# training_dataset_img_paths = training_dataset[1]
# training_dataset_target_paths = training_dataset[2]

# Validation set
validation_img_dir = "data/img/val"
validation_mask_dir = "data/masks/val"
validation_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=validation_img_dir,
    target_img_dir=validation_mask_dir,
)
# validation_dataset_batch = validation_dataset[0]
# validation_dataset_img_paths = validation_dataset[1]
# validation_dataset_target_paths = validation_dataset[2]


# Test set
test_img_dir = "data/img/test"
test_mask_dir = "data/masks/test"
test_dataset = pipeline.get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=test_img_dir,
    target_img_dir=test_mask_dir,
)
# test_dataset_batch = test_dataset[0]
# test_dataset_img_paths = test_dataset[1]
# test_dataset_target_paths = test_dataset[2]


# Creates the model itself
model = models.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
)

# Callback for saving weights
CHECKPOINT_FILEPATH = "./models/model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
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

print("---------------------------------------------------------------------------------------------------")

print("Predictions: ")

predictions = model.predict(test_dataset)

print(predictions.shape)

mask = np.argmax(predictions[1], axis=-1)

# Flatten the predicted_mask to make it a 1D array, since we're interested in the global distribution
flat_predicted_mask = mask.flatten()

# Get unique classes and their counts
classes, counts = np.unique(flat_predicted_mask, return_counts=True)

# To see the distribution, you can print it or store it in a dictionary
class_distribution = dict(zip(classes, counts))
print("Class distribution in the predicted output:", class_distribution)