# from cv2 import applyColorMap
import numpy as np
import tensorflow as tf
<<<<<<< HEAD
from tensorflow import keras
=======
import keras
import os
import sys
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709
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
<<<<<<< HEAD
BATCH_SIZE = 32
EPOCHS = 10
=======
BATCH_SIZE = 8
EPOCHS = 2
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709

# * Datasets

# Trainig set
training_pipeline = Pipeline()
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_pipeline.set_dataset_from_directory(
    training_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
<<<<<<< HEAD
)
# training_dataset_batch = training_dataset[0]
<<<<<<<< HEAD:train_model.py
# training_dataset_img_paths = training_dataset[1]
========
# training_dataset_img_paths = training_dataset[2]
>>>>>>>> bef75139884aa5ddc38a13c8e9de2bc8b5d43959:scripts/train_model.py
# training_dataset_target_paths = training_dataset[2]
=======
    batch_size=BATCH_SIZE,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
training_dataset = training_pipeline.dataset
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709

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
<<<<<<< HEAD
<<<<<<<< HEAD:train_model.py
# test_dataset_batch = test_dataset[0]
# test_dataset_img_paths = test_dataset[1]
# test_dataset_target_paths = test_dataset[2]
========
# test_dataset_batch = training_dataset[0]
# test_dataset_img_paths = training_dataset[1]
# test_dataset_target_paths = training_dataset[2]
>>>>>>>> bef75139884aa5ddc38a13c8e9de2bc8b5d43959:scripts/train_model.py

=======
test_dataset = test_pipeline.dataset
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709

# Creates the model itself
model = model_architectures.get_UNET_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
)

# Callback for saving weights
<<<<<<< HEAD
CHECKPOINT_FILEPATH = "./models/model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
=======
#TODO add model name to enviroment upon calling job
CHECKPOINT_FILEPATH = f"./models/{os.environ.get("MODEL_NAME")}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    mode="max",
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709
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

<<<<<<<< HEAD:train_model.py
print("Predictions: ")

predictions = model.predict(test_dataset)

print(predictions.shape)
========
print(predictions)
>>>>>>>> bef75139884aa5ddc38a13c8e9de2bc8b5d43959:scripts/train_model.py

mask = np.argmax(predictions[1], axis=-1)

<<<<<<< HEAD
# Flatten the predicted_mask to make it a 1D array, since we're interested in the global distribution
flat_predicted_mask = mask.flatten()

# Get unique classes and their counts
classes, counts = np.unique(flat_predicted_mask, return_counts=True)

# To see the distribution, you can print it or store it in a dictionary
class_distribution = dict(zip(classes, counts))
<<<<<<<< HEAD:train_model.py
print("Class distribution in the predicted output:", class_distribution)
========
print("Class distribution in the predicted output:", class_distribution)
>>>>>>>> bef75139884aa5ddc38a13c8e9de2bc8b5d43959:scripts/train_model.py
=======
# TODO Predict whole test set on mIOU metric and make a custom output
>>>>>>> ccafbd41765595192d4157d0ee8859715f790709
