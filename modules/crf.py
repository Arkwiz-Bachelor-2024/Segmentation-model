import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

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
        sdims=(8, 8), schan=(13, 13, 13), img=image, chdim=2
    )

    # ? Compat - The potentials influence on the final labeling. The strength of the potential.
    d.addPairwiseEnergy(gaussian_potential, compat=3)
    d.addPairwiseEnergy(bilateral_potential, compat=5)

    # Perform inference to get the refined segmentation.
    Q = d.inference(inference_iterations)

    # `map_soln` is now your refined segmentation mask.
    crf_mask = np.argmax(Q, axis=0).reshape(image.shape[:2])

    return crf_mask
