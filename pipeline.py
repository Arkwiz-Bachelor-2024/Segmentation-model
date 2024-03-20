"""
A pipeline which is able to create a dataset based upon specified image directories or paths

Code inspired by:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""

import os
from natsort import natsorted

import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import io as tf_io
from tensorflow import image as tf_image


def __decode_dataset__(input_img_path, target_img_path):
    """
    Rescales,resizes and decodes an image and its respective mask to a tensor.

    """
    # Converts the input image to a tensor
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_jpeg(input_img, channels=3)
    input_img = tf_image.resize(input_img, (512, 512))
    input_img = tf.cast(input_img, tf.float32) / 255.0
    # Dtype describes how the bytes of the image are to be interpeted, e.g the format of the image.
    input_img = tf_image.convert_image_dtype(input_img, "float32")

    # Converts the respective target image(mask) to a tensor
    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf.cast(target_img, tf.float32) / 255.0
    target_img = tf_image.resize(target_img, (512, 512), method="nearest")
    target_img = tf_image.convert_image_dtype(target_img, "uint8")

    return input_img, target_img


def get_dataset_from_paths(
    batch_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """
    Returns an array sequentially containing the datset batch of given batch size based upon given paths along with
    the path of the input and target images.

    """

    # For faster debugging, limits the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(__decode_dataset__, num_parallel_calls=tf_data.AUTOTUNE)

    return dataset.batch(batch_size)


def get_dataset_from_directory(
    batch_size,
    input_img_dir,
    target_img_dir,
    max_dataset_len=None,
):
    """
    Returns a datset batch of given batch size extracted from a specified directory.

    """

    input_img_paths = __sort_directory__(input_img_dir)
    target_img_paths = __sort_directory__(target_img_dir)

    # For faster debugging, limits the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(__decode_dataset__, num_parallel_calls=tf_data.AUTOTUNE)

    return dataset.batch(batch_size)


def __sort_directory__(input_dir):
    """
    Extracts files from a directory into a naturally sorted array.

    """

    input_files = natsorted(
        [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
    )

    return input_files
