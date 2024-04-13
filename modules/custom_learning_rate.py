import tensorflow as tf

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, peak_lr, warmup_batches, switch_epoch, schedule):
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        self.switch_epoch = switch_epoch
        self.switched = False
        self.schedule = schedule

    def on_batch_begin(self, batch, logs=None):
    
        if self.batch_count <= self.warmup_batches:  # Warm-up phase
            lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (self.batch_count / self.warmup_batches)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        self.batch_count += 1

    def on_epoch_end(self, epoch, logs=None):
        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate

        # Swtich to SGD
        if epoch >= self.switch_epoch and not self.switched:  
            self.model.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, weight_decay=0.0001, clipnorm=0.5, momentum=0.9)
            self.switched = True
            print(f"Switched to SGD at epoch {epoch}")
            
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
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

            print(f"\nEpoch {epoch}: Learning rate is {float(lr_value):.6}.")

