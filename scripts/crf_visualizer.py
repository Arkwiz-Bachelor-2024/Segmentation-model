import sys
import os

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from modules.pipeline import Pipeline
from modules.crf import custom_conditional_random_field, crf_mask_grid_search
from modules.plot import simple_image_display

# * Components
<<<<<<< HEAD
model = keras.models.load_model("./models/seg_model_10e_64b_+DA")
=======
model = keras.models.load_model("./models/10epoch_32b.keras", compile=False)
>>>>>>> c541a8674be407055254303d9a815e240b31460b
pipeline = Pipeline()
pipeline.set_dataset_from_directory(
    batch_size=1,
    input_img_dir="data/img/test",
    target_img_dir="data/masks/test",
)
dataset = pipeline.dataset

# White, Red, Green, Blue, Gray
colors = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5)]
cmap = ListedColormap(colors)

# even_images_indecies = [537, 1014, 1190, 71, 84, 1305, 1215, 86, 1184, 547]
even_images_indecies = [537, 1014, 1190, 71, 84, 1305, 1215, 86, 1184, 547]

images = []
masks = []
for index in even_images_indecies:
    sample = pipeline.get_sample_by_index(index, 1)
    images.append(sample[0])
    masks.append(sample[1])

pred_mask_probs_list = []
for image in images:
    # image_with_batch = np.expand_dims(image, axis=0)
    pred_mask_probs_list.append(model.predict(image))

# TODO refactor to different sdims for each potential
sdims_options = [(2, 2), (1, 1), (3, 3)]
compats_options = [(5, 5)]

crf_grid_details = crf_mask_grid_search(
    images=images,
    pred_mask_probs_list=pred_mask_probs_list,
    masks=masks,
    compats_options=compats_options,
    sdims_options=sdims_options,
)

crf_grid_masks = [detail["mask"] for detail in crf_grid_details]
crf_grid_scores = [detail["score"] for detail in crf_grid_details]

crf_grid_parameters = [detail["parameters"] for detail in crf_grid_details]
crf_grid_sdims = [detail["sdims"] for detail in crf_grid_parameters]
crf_grid_compat_g = [detail["compat_gaussian"] for detail in crf_grid_parameters]
crf_grid_compat_b = [detail["compat_bilateral"] for detail in crf_grid_parameters]

crf_titles = []
for i in range(len(crf_grid_parameters)):
    title = crf_grid_sdims[i] + " " + crf_grid_compat_g[i] + " " + crf_grid_compat_b[i]

    crf_titles.append(title)

index = 2

crf_grid_masks.append(images[index].numpy().squeeze())
crf_grid_scores.append(" ")
crf_titles.append("Preview Image")

crf_grid_masks.append(masks[index].numpy().squeeze())
crf_grid_scores.append(" ")
crf_titles.append("Preview Mask")

crf_grid_masks.insert(0, pred_mask_probs_list[index].argmax(axis=-1).squeeze())
crf_grid_scores.insert(0, " ")
crf_titles.insert(0, "Prediction")

print(crf_grid_scores)
print(crf_grid_parameters)
print(crf_grid_sdims)
print(crf_grid_compat_g)
print(crf_grid_compat_b)
print(crf_titles)


simple_image_display(
    images=crf_grid_masks,
    titles=crf_titles,
    descriptions=crf_grid_scores,
    color_map=cmap,
)
