import tensorflow as tf
from tensorflow import keras
import numpy as np


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).


    Extracted from: https://keras.io/api/callbacks/learning_rate_scheduler/
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        self.model.optimizer.learning_rate = scheduled_lr

        if epoch > 0:
            # Check if scheduled_lr is a MirroredVariable and read its value
            if isinstance(scheduled_lr, tf.distribute.DistributedValues):
                # Read the value of the variable and convert it to a numpy array
                lr_value = scheduled_lr.read_value().numpy()
            else:
                # If not a DistributedValues type, try to directly convert to float
                lr_value = float(scheduled_lr)

            print(f"\nEpoch {epoch}: Learning rate is {float(lr_value):.6}.")
