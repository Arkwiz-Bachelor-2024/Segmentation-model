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


# TODO Refactor dataset specifications for channels, shape, batch size etc.
class Pipeline:

    def __init__(self):
        self.input_img_paths = None
        self.target_img_paths = None
        self.dataset = None
        self.batch_size = None

    def get_sample_by_filename(self, filename):
        index = next(
            i for i, path in enumerate(self.input_img_paths) if filename in path
        )
        sample = self.dataset.skip(index).take(1)
        for image, mask in sample:
            return image, mask

    def get_sample_by_index(self, sample_index, batch_size):
        # if batch_size == 1:
        #     return self.dataset.skip(sample_index).take(1)

        # # Calculate which batch the sample is in
        # batch_index = sample_index // batch_size
        # # Calculate the index of the sample within its batch
        # index_within_batch = sample_index % batch_size

        # Extract the batch
        for image, mask in self.dataset.skip(sample_index).take(1):
            image = image
            mask = mask

        return image, mask

    def set_dataset_from_directory(
        self,
        batch_size,
        input_img_dir,
        target_img_dir,
        max_dataset_len=None,
        per_class_masks=False,
    ):
        """
        Returns a datset batch of given batch size extracted from a specified directory.

        """

        input_img_paths = self.__sort_directory__(input_img_dir)
        target_img_paths = self.__sort_directory__(target_img_dir)

        # Save details to object
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.batch_size = batch_size

        # For faster debugging, limits the size of data
        if max_dataset_len:
            input_img_paths = input_img_paths[:max_dataset_len]
            target_img_paths = target_img_paths[:max_dataset_len]

        dataset = tf_data.Dataset.from_tensor_slices(
            (input_img_paths, target_img_paths)
        )
        if per_class_masks:
            dataset = dataset.map(
                self.__decode_dataset_multi_loss__, num_parallel_calls=tf_data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                self.__decode_dataset__, num_parallel_calls=tf_data.AUTOTUNE
            )

        self.dataset = dataset.batch(batch_size)

    def __sort_directory__(self, input_dir):
        """
        Extracts files from a directory into a naturally sorted array.

        """

        input_files = natsorted(
            [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
        )

        return input_files

    def __decode_dataset__(self, input_img_path, target_img_path):
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

        # Data augmentation which will be applied diffrently each epoch giving different versions of the images each time.
        # input_img, target_img = self.__augment_image__(input_img, target_img)

        return input_img, target_img

    def __decode_dataset_multi_loss__(self, input_img_path, target_img_path):
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

        num_classes = 5

        target_img_one_hot = tf.one_hot(target_img, depth=num_classes)
        # One hot encodes to mask
        # e.g post_target_img_one_hot:  [[[[0 0 1 0 0]]
        # 512 arrays with 512 elements each with a one-hot encoded array for the pixel
        # Which can be sliced in order to obtain the mask for each class

        # Remove old dimension
        target_img_one_hot = tf.squeeze(target_img_one_hot, axis=-2)

        # Data augmentation which will be applied differently each epoch giving different versions of the images each time.
        input_img, target_img_one_hot = self.__augment_image__(
            input_img, target_img_one_hot
        )

        return input_img, target_img_one_hot

    def __augment_image__(self, input_img, target_img):
        """
        Data augmentation applied to the image in the form of:

        - Flipping the image
        - Rotations
        - Brightness,contrast,hue
        - Gaussian noise

        """

        # Combined random flip horizontally
        if tf.random.uniform(()) > 0.5:
            input_img = tf.image.flip_left_right(input_img)
            target_img = tf.image.flip_left_right(target_img)

        # Combined random flip vertically
        if tf.random.uniform(()) > 0.5:
            input_img = tf.image.flip_up_down(input_img)
            target_img = tf.image.flip_up_down(target_img)

        # Random rotation
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        input_img = tf.image.rot90(input_img, k=k)
        target_img = tf.image.rot90(target_img, k=k)

        # Adjust brightness (photometric transformation)
        input_img = tf.image.random_brightness(input_img, max_delta=0.1)

        # Adjust contrast (photometric transformation)
        input_img = tf.image.random_contrast(input_img, lower=0.9, upper=1.1)

        # Adjust hue (photometric transformation)
        input_img = tf.image.random_hue(input_img, max_delta=0.1)

        # Adjust saturation (photometric transformation)
        input_img = tf.image.random_saturation(input_img, lower=0.9, upper=1.1)

        # Adding Gaussian noise (noise injection)
        noise = tf.random.normal(
            shape=tf.shape(input_img), mean=0.0, stddev=0.1, dtype=tf.float32
        )
        input_img = input_img + noise
        input_img = tf.clip_by_value(
            input_img, 0.0, 1.0
        )  # Ensure values are still in [0, 1] range

        return input_img, target_img
