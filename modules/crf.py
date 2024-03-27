import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

# Heuristic approach

# Parameters
# sdim - Dimensions of filter
# schan - Impact of each channel(colors)
# compat - Weight attributed to each potential in the context of labeling
# Inference - Iterations of labeling based on potentials

#TODO sort dataset by uniform class distribution
# Binary grid search
# Dataset 
# 10 pictures with most uniform class distribution from test set predicted

# sdim, schan, compat
# 5 inference iterations as an estimate to show the impact of crf, might try 3
# 3 models for each parameter with average and upper and lower
# Upon reaching a new best model, take the average between bounds and run again


# Assumptions for starting point of potentials
# Rough estimate of smallest objects to identify meaning the dimensons of the potential impact
# has to be able to take into account objects of this size
# - Road width around 2 - 4 meters : 4 to 8 pixels
# - Tree width from 2 - 10 meters : 4 to 20 pixels


def conditional_random_field(image, pred_mask_probs, inference_iterations):

    # Assuming `image` is your input image in [0, 255] and `pred_mask_logits` are the logits
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], pred_mask_probs.shape[-1])

    # Move classes first because its needed for some reason
    pred_mask_probs = np.transpose(pred_mask_probs.squeeze(), (2, 0, 1))
    pred_mask_probs = np.ascontiguousarray(pred_mask_probs)

    # The -log(p) of the pixel values
    U = unary_from_softmax(pred_mask_probs)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian potentials for encouraging nearby pixels with similar color to get similar labels.
    # Favours clustering of pixels
    # ? sdims - Kernel/filter size when calculating potential. Spatial proximity.
    gaussian_potential = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])

    # Pairwise Bilateral potentials for encouraging nearby pixels with similar color and intensity to get similar labels.
    # Favours establishing edges
    # ? schan - The influence of color differences when labeling.
    bilateral_potential = create_pairwise_bilateral(
        sdims=(80, 80), schan=(13, 13, 13), img=image, chdim=2
    )

    # ? Compat - The potentials influence on the final labeling. The strength of the potential.
    d.addPairwiseEnergy(gaussian_potential, compat=3)
    d.addPairwiseEnergy(bilateral_potential, compat=10)

    # Perform inference to get the refined segmentation.
    Q = d.inference(inference_iterations)

    # `map_soln` is now your refined segmentation mask.
    crf_mask = np.argmax(Q, axis=0).reshape(image.shape[:2])

    return crf_mask
