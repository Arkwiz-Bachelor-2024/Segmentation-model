from keras import backend as K

"""
Module containing custom loss functions used with Keras API.
"""


# Smooth is added to avoid dividing by 1
def __tversky_index_class__(class_id, y_true, y_pred, alpha, beta, smooth=100):
    """
    Calculates the tvernsky index for a given class id.
    """

    # One-hot encode and cast to float32
    y_true = K.cast(K.equal(y_true, class_id), "float32")

    # y_true = K.print_tensor(y_true, message="y_true = ", summarize=-1)
    # y_pred = K.print_tensor(y_pred, message="y_pred = ", summarize=-1)

    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))
    return (
        2 * (true_positives + smooth)
        / (true_positives + alpha * false_positives + beta * false_negatives + smooth)
    )


def multi_class_tversky_loss(weights):
    """
    Calculates the total loss based on pre-determined weights for each class.
    Expects a list of tuples corresponding to the weights of its respective class
    Format: (FP, FN) for each class
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
            tversky_loss += __tversky_index_class__(
                class_idx,
                # ... slices all other axies before the ... ,in this case for each time in the for loop it will
                # "slice" out a 2D array containing predictions on that class
                y_true[..., class_idx],
                y_pred[..., class_idx],
                alpha,
                beta,
            )

        return 1 - (tversky_loss / num_classes)

    return loss
