"""
Module which offers ways to visualize the datset and the predictions made by the model.

"""

import matplotlib.colors as mcolors
from cv2 import applyColorMap
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# Custom color map for labels
COLOR_MAP = {
    0: "rgb(0,0,0)",  # Black for background
    1: "rgb(220,220,220)",  # Light grey for buildings
    2: "rgb(0,255,0)",  # Green for forests
    3: "rgb(0,0,255)",  # Blue for water
    4: "rgb(128,128,128)",  # Grey for roads
}


def visualize(mask, image):

    # Assuming `image` is your original image and `mask` is the segmentation mask
    # Apply the color map to the mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Apply color map

    # Initialize an empty RGB image with the same height and width as the mask
    for label, color in COLOR_MAP.items():
        colored_mask[mask == label] = np.array(
            color[4:-1].split(","), dtype=np.uint8
        )  # Convert color to RGB values

    colored_mask = applyColorMap(mask, COLOR_MAP)

    # Create subplots
    fig = go.Figure()

    # Add the original image
    fig.add_trace(go.Image(z=image))

    # Add the color-mapped mask
    fig.add_trace(go.Image(z=colored_mask))

    # Update layout for a side-by-side comparison
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=1.15,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Show Original",
                        method="update",
                        args=[{"visible": [True, False]}],
                    ),
                    dict(
                        label="Show Mask",
                        method="update",
                        args=[{"visible": [False, True]}],
                    ),
                ],
            )
        ]
    )

    # Show figure
    fig.show()


# Define your custom colors
label_colors = [
    (0, 0, 0),  # 0: Black
    (
        0.75,
        0.75,
        0.75,
    ),  # 1: Light gray (for buildings, more distinguishable than white)
    (0, 0.5, 0),  # 2: Green (darker tone for forests)
    (0, 0, 1),  # 3: Blue
    (0.5, 0.5, 0.5),  # 4: Grey
]

# Create a ListedColormap
custom_cmap = mcolors.ListedColormap(label_colors)

# # Custom color map for labels
# COLOR_MAP = {
#     0: "rgb(0,0,0)",  # Black for background
#     1: "rgb(220,220,220)",  # Light grey for buildings
#     2: "rgb(0,255,0)",  # Green for forests
#     3: "rgb(0,0,255)",  # Blue for water
#     4: "rgb(128,128,128)",  # Grey for roads
# }


# def apply_custom_colormap(mask, color_map):
#     # Initialize an RGB image with the same dimensions as the mask
#     colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     for label, color_str in color_map.items():
#         color = [int(c) for c in color_str.replace('rgb(', '').replace(')', '').split(',')]
#         colored_mask[mask == label] = color
#     return colored_mask


# Assuming `dataset` is the final dataset object after applying all preprocessing steps

# # Take a single batch from the dataset
# for input_images, target_masks in test_dataset.skip(500).take(8):

#     # Print shapes, data types, and some statistics for the batch
#     print(f"Input Image Batch Shape: {input_images.shape}")
#     print(f"Input Image Data Type: {input_images.dtype}")
#     print(f"Min Value in Input Images: {np.min(input_images)}")
#     print(f"Max Value in Input Images: {np.max(input_images)}\n")

#     print(f"Target Mask Batch Shape: {target_masks.shape}")
#     print(f"Target Mask Data Type: {target_masks.dtype}")
#     print(f"Unique Values in Target Masks: {np.unique(target_masks)}\n")

#     # Optionally, visualize the first image and mask from the batch to check correctness
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(input_images[0].numpy())
#     plt.title("Input Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(
#         target_masks[0].numpy().squeeze(), cmap=custom_cmap
#     )  # Adjust as necessary for mask dimensions
#     plt.colorbar()
#     plt.title("Target Mask")
#     plt.axis("off")

#     plt.show()
