import tensorflow as tf
import numpy as np

# Placeholder for evenness scores and mask indices
evenness_scores = []
mask_indices = []

# Ideal distribution: each class having exactly 1/5 of the total pixels
ideal_distribution = np.ones(5) / 5

# Loop through the dataset
for i, (image, mask) in enumerate(dataset):
    # Calculate the distribution of classes in the current mask
    # mask should be a 2D tensor/array with class labels (0 to 4) for each pixel
    class_counts = np.bincount(mask.flatten(), minlength=5)
    total_pixels = mask.size
    class_distribution = class_counts / total_pixels

    # Calculate the evenness score as the sum of absolute differences from the ideal distribution
    evenness_score = np.sum(np.abs(class_distribution - ideal_distribution))

    # Append the score and index to the lists
    evenness_scores.append(evenness_score)
    mask_indices.append(i)

# After looping through all masks, select the top 10 masks with the most even distribution
top_10_indices = np.argsort(evenness_scores)[:10]

# Print or return the indices of the top 10 masks
print("Top 10 Masks by Evenness:", top_10_indices)
