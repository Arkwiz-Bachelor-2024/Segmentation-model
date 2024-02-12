"""
A script which creates and returns a dataset based upon specified image directories

Code inspired by:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""

import os
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

# * Masks and image directories
input_dir = "img/"
target_dir = "masks/"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:20], target_img_paths[:20]):
    print(input_path, "|", target_path)


def load_img_masks(input_img_path, target_img_path, img_size):
    """
    Converts an image and its respective mask to a tensor.
    """

    # Converts the input image to a tensor
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_jpeg(input_img, channels=3)
    input_img = tf_image.resize(input_img, img_size)
    # Dtype describes how the bytes of the image are to be interpeted, e.g the format of the image.
    input_img = tf_image.convert_image_dtype(input_img, "float32")

    # Converts the respective target image(mask) to a tensor
    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf_image.resize(target_img, img_size, method="nearest")
    target_img = tf_image.convert_image_dtype(target_img, "uint8")

    # Ground truth labels are 1, 2, 3. Subtracts one to make them 0, 1, 2:
    target_img -= 1
    return input_img, target_img


"""
Returns a TF Dataset batch of given batch size

"""


def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)

    return dataset.batch(batch_size)
