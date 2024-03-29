from keras import backend as K

"""
Module containing custom loss functions used with Keras API.
"""


# Smooth is added to avoid dividing by 0
def __tversky_index__(y_true, y_pred, alpha, beta, smooth=1e-6):
    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))
    return (true_positives + smooth) / (
        true_positives + alpha * false_positives + beta * false_negatives + smooth
    )


def multi_class_tversky_loss(weights):
    """
    Calculates the total loss based on pre-determined weights for each class.
    Expects a list of tuples corresponding to the weights of its respective class
    e.g [(1,1), (0.7,0.8)......]

    """

    def loss(y_true, y_pred):
        tversky_loss = 0.0
        num_classes = K.int_shape(y_pred)[-1]

        # Ensure the weights list matches the number of classes
        assert (
            len(weights) == num_classes
        ), "Weights list must match the number of classes."

        for class_idx in range(num_classes):
            alpha, beta = weights[class_idx]
            tversky_loss += __tversky_index__(
                # ... slices all other axies in this case for each time in the for loop it will
                # "slice" out a 2D array containing predictions on that class
                y_true[..., class_idx],
                y_pred[..., class_idx],
                alpha,
                beta,
            )

        return tversky_loss / num_classes

    return loss
