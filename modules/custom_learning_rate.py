import tensorflow as tf


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        base_lr,
        initial_lr,
        warmup_batches,
        max_epochs,
        schedule,
    ):
        self.base_lr = base_lr
        self.initial_lr = initial_lr
        self.warmup_batches = warmup_batches
        self.max_epochs = max_epochs
        self.schedule = schedule
        self.batch_count = 0

    def on_batch_begin(self, batch, logs=None):

        # Warmup learning rate schedule
        if self.batch_count <= self.warmup_batches:
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * (
                self.batch_count / self.warmup_batches
            )
            self.model.optimizer.learning_rate = lr

        self.batch_count += 1

    def on_epoch_end(self, epoch, logs=None):

        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate

        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.base_lr, self.max_epochs)
        self.model.optimizer.learning_rate = scheduled_lr

        # Check if value is of type float or MirroredVaraible when using multiple GPUs
        if epoch > 0:
            # Check if scheduled_lr is a MirroredVariable and read its value
            if isinstance(scheduled_lr, tf.distribute.DistributedValues):
                # Read the value of the variable and convert it to a numpy array
                lr_value = scheduled_lr.read_value().numpy()
            else:
                # If not a DistributedValues type, try to directly convert to float
                lr_value = float(scheduled_lr)

            print(f"\nEpoch {epoch}: Learning rate was {float(lr_value):.6}.")
