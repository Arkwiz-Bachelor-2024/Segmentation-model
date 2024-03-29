import matplotlib.pyplot as plt
import numpy as np


def simple_image_display(titles, images, descriptions, color_map):
    """
    displays images with their individual titles and array of descriptions.

    parameters:
    - titles: list of str, titles for each image.
    - images: list of ndarray, image data in a format compatible with plt.imshow.
    - descriptions: list of list of str, each list contains parts of the description for each image.
    """
    if not images or not descriptions or not titles:
        raise valueerror("titles, images, and descriptions must be non-empty lists.")

    if len(images) != len(descriptions) or len(images) != len(titles):
        raise valueerror(
            "titles, images, and descriptions must be lists of the same length."
        )

    num_images = len(images)
    plt.figure(figsize=(15, 8))

    for i in range(num_images):
        plt.subplot(2, 2, i + 1)
        # Shortcut
        if images[i].shape[-1] > 10:
            plt.imshow(images[i], cmap=color_map)
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

        # Join the array of details into a single string with line breaks.
        description_text = "\n".join(descriptions[i])

        # Adding the description text below the image
        plt.text(
            0.5,
            -0.1,
            description_text,
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots a confusion matrix using Matplotlib's functionality.

    Args:
    - cm: The confusion matrix to plot.
    - class_names: An array of labels for the classes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".1f"  # Format for annotations inside the heatmap.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.title("Normalized Confusion Matrix")
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
