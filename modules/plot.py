import matplotlib.pyplot as plt
import numpy as np
import math


def simple_image_display(titles=None, images=None, descriptions=None, color_map=None):
    """
    displays images with their individual titles and array of descriptions.

    parameters:
    - titles: list of str, titles for each image.
    - images: list of ndarray, image data in a format compatible with plt.imshow.
    - descriptions: list of list of str, each list contains parts of the description for each image.
    - colour_map : str, colour map to be used for displaying the image.
    """

    num_images = len(images)
    plt.figure(figsize=(15, 8))

    # Calculate the grid scale based on the number of images.
    grid_scale = math.ceil(math.sqrt(num_images))
    for i in range(num_images):
        plt.subplot(grid_scale, grid_scale, i + 1)
        # Shortcut
        if images[i].shape[-1] > 10:
            plt.imshow(images[i], cmap=color_map)
        else:
            plt.imshow(images[i])

        if titles != None:
            plt.title(titles[i])
        plt.axis("off")

        if descriptions != None:
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


def plot_confusion_matrix(cm, class_names, save_path, title):
    """
    Plots a confusion matrix using Matplotlib's functionality.

    Args:
    - cm: The confusion matrix to plot.
    - class_names: An array of labels for the classes.
    - save_path: The path to save the plot.
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

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".1f"
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
    plt.title(title)
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
