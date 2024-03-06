import tensorflow as tf

"""
Script for checking if GPU's are being used to train the model.

"""

# Check available GPUs
gpus = tf.config.list_physical_devices("GPU")

# Print the list of available GPUs
print("GPUs Available: ", gpus)
