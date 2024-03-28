import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)
from modules.metrics import get_mIOU


def pre_defined_conditional_random_field(image, pred_mask_probs, inference_iterations):

    # Assuming `image` is your input image in [0, 255] and `pred_mask_logits` are the logits
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], pred_mask_probs.shape[-1])

    # Move classes first because its needed for some reason
    pred_mask_probs = np.transpose(pred_mask_probs.squeeze(), (2, 0, 1))
    pred_mask_probs = np.ascontiguousarray(pred_mask_probs)

    # The -log(p) of the pixel values
    U = unary_from_softmax(pred_mask_probs)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian potentials for encouraging nearby pixels with similar color to get similar labels.
    # ? sdims - Distance in each axis where pixels can influence labeling
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

    crf_mask = np.argmax(Q, axis=0).reshape(image.shape[:2])

    return crf_mask


def custom_conditional_random_field(image, pred_mask_probs, sdim, schan, compat):

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], pred_mask_probs.shape[-1])
    pred_mask_probs = np.transpose(pred_mask_probs.squeeze(), (2, 0, 1))
    pred_mask_probs = np.ascontiguousarray(pred_mask_probs)

    U = unary_from_softmax(pred_mask_probs)
    d.setUnaryEnergy(U)

    gaussian_potential = create_pairwise_gaussian(sdims=sdim, shape=image.shape[:2])
    bilateral_potential = create_pairwise_bilateral(
        sdims=sdim, schan=schan, img=image, chdim=2
    )

    d.addPairwiseEnergy(gaussian_potential, compat=compat[0])
    d.addPairwiseEnergy(bilateral_potential, compat=compat[1])

    Q = d.inference(5)

    crf_mask = np.argmax(Q, axis=0).reshape(image.shape[:2])

    return crf_mask


def crf_mask_grid_search(
    pred_mask_probs_list, images, masks, sdims_options, compats_options
):

    # Assuming `images` and `pred_mask_probs_list` are defined, and `true_masks` for evaluation
    # Each element in `images` and `pred_mask_probs_list` corresponds to one image and its predicted mask probabilities
    results = []  # List to store parameter sets with their scores

    for sdims in sdims_options:
        for compats in compats_options:
            compat_gaussian, compat_bilateral = compats
            crf_masks = []  # Store masks generated with the current parameter set

            # Generate 10 CRF masks for the current parameter combination
            for image, pred_mask_probs in zip(images, pred_mask_probs_list):
                crf_mask = custom_conditional_random_field(
                    image=image.numpy(),
                    pred_mask_probs=pred_mask_probs,
                    sdim=sdims,
                    schan=(10, 10, 10),
                    compat=compats,
                )
                crf_masks.append(crf_mask)

            # Avg mIOU
            scores = []
            for mask, crf_mask in zip(masks, crf_masks):
                scores.append(get_mIOU(mask, crf_mask, 5))

            score = sum(scores) / len(scores)

            # Store the parameter set and its score
            results.append(
                {
                    "parameters": {
                        "sdims": sdims,
                        "compat_gaussian": compat_gaussian,
                        "compat_bilateral": compat_bilateral,
                    },
                    "mask": crf_masks[0],
                    "score": score,
                }
            )

    # Sort the results by score to find the best-performing parameter sets
    return sorted(results, key=lambda x: x["score"], reverse=True)
