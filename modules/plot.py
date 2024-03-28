import matplotlib.pyplot as plt


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
