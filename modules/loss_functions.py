import tensorflow as tf
from tensorflow import keras
from keras import backend as K

"""
Module containing custom loss functions used with Keras API.
"""


# Smooth is added for numerical stability
def __tversky_index_class__(y_true, y_pred, alpha, beta, smooth=1e-5):
    """
    Calculates the tvernsky index for a given class id.
    """

    # Calculates the number of true positives, false positives and false negatives
    # This works cause the one-hot encoding ensures that the multiplication of the arrays will
    # only result in a positive value if the prediction is correct
    # e.g [0,0,1,0] * [1,0,1,0] = [0,0,1,0] = 1 as only on of them is actually correct

    # tf.print("tvernsky_y_true: ", y_true)
    # tf.print("tvernsky_y_pred: ", y_pred)
    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))

    return (true_positives + smooth) / (
        true_positives + alpha * false_positives + beta * false_negatives + smooth
    )


def multi_class_loss(tvernsky_weights, cross_entropy_weights, DEBUG=False):
    """
    #TODO document
    Calculates the total loss based on pre-determined weights for each class.
    Expects a list of tuples corresponding to the weights of its respective class
    Format: (FP, FN) for each class
    e.g [(1,1), (0.7,0.8)......]


    """

    def loss(y_true, y_pred):
        """
        Computes the sum of the loss on the local batch

        """
        num_classes = K.int_shape(y_pred)[-1]

        tversky_loss = 0.0
        binary_cross_entropy_loss = 0.0

        # Ensure the weights list matches the number of classes
        assert (
            len(tvernsky_weights) and len(cross_entropy_weights) == num_classes
        ), "Weights list must match the number of classes."

        for class_idx in range(num_classes):

            y_pred_sliced = y_pred[..., class_idx]
            # y_pred_sliced = tf.expand_dims(y_pred, axis=-1)
            # Y_pred is sliced so that it contains an array of the probabilities for each pixel
            # that it belongs to this class
            # An array of 512 rows where each row contains 512 probabilities corresponding to each pixel
            # e.g [0.3,0,4,0.1,0,3........0.1,.2]
            #     ....
            # 512x ....
            #     ....
            #     [0.3,0,4,0.1,0,3........0.1,.2]

            y_true_sliced = y_true[..., class_idx]
            # y_true_sliced = tf.expand_dims(y_true, axis=-1)
            # y_true is the same aswell only that the values are 0 or 1

            # Compute tversnky loss
            alpha_tversnky, beta_tvernsky = tvernsky_weights[class_idx]
            tversky_loss += __tversky_index_class__(
                # ... slices all other axies before the ... ,in this case for each time in the for loop it will
                # "slice" out a 2D array containing predictions on that class
                y_true_sliced,
                y_pred_sliced,
                alpha_tversnky,
                beta_tvernsky,
            )

            # Compute cross entropy loss
            class_weight = cross_entropy_weights[class_idx]
            binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM,
                from_logits=False,
            )
            pre_weighted_loss = binary_cross_entropy(y_true_sliced, y_pred_sliced)
            binary_cross_entropy_loss += class_weight * pre_weighted_loss

        total_tvernsky_loss = 1 - tversky_loss / num_classes
        total_binary_cross_entropy_loss = binary_cross_entropy_loss / num_classes
        total_local_batch_loss = total_tvernsky_loss + total_binary_cross_entropy_loss

        if DEBUG:
            tf.print("---------------------------")
            tf.print("Loss for this batch:")
            tf.print("Binary cross entropy loss: ", binary_cross_entropy_loss)
            tf.print("Tversky loss: ", tversky_loss)
            tf.print("Total loss: ", total_local_batch_loss)
            tf.print("Total tversky loss: ", total_tvernsky_loss)
            tf.print(
                "Total_binary_cross_entropy_loss: ", total_binary_cross_entropy_loss
            )

        return total_local_batch_loss

    return loss
