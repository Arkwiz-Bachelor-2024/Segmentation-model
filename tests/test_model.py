import keras
import sys
import os
import numpy as np

# Imports the root directory based on the location of this file.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from pipeline import get_dataset_from_directory

# Path to your saved model
model_path = "./models/checkpoint.model.keras"

# Load the model
model = keras.models.load_model(model_path)

# * Datasets

BATCH_SIZE = 4
MAX_NUMBER_SAMPLES = 20

# Trainig set
training_img_dir = "data/img/train"
training_mask_dir = "data/masks/train"
training_dataset = get_dataset_from_directory(
    batch_size=BATCH_SIZE,
    input_img_dir=training_img_dir,
    target_img_dir=training_mask_dir,
    max_dataset_len=MAX_NUMBER_SAMPLES,
)
training_dataset_batch = training_dataset[0]
training_dataset_img_paths = training_dataset[2]
training_dataset_target_paths = training_dataset[2]


predictions = model.predict(training_dataset_batch)

print(predictions.shape)

mask = np.argmax(predictions[0], axis=-1)

# Flatten the predicted_mask to make it a 1D array, since we're interested in the global distribution
flat_predicted_mask = mask.flatten()

# Get unique classes and their counts
classes, counts = np.unique(flat_predicted_mask, return_counts=True)

# To see the distribution, you can print it or store it in a dictionary
class_distribution = dict(zip(classes, counts))
print("Class distribution in the predicted output:", class_distribution)
