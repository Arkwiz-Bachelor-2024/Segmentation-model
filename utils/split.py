#!/usr/bin/env python3
"""
Utility script which takes input images and masks in a .tiff formats them into a specified resoulution images in .jpg and .png format respectively.

Script is inspired by: https://landcover.ai.linuxpolska.com/
"""

import glob
import os
import cv2

# Parameters for conversion
IMGS_DIR = "./images"
MASKS_DIR = "./masks"
OUTPUT_DIR = "./output"
TARGET_SIZE = 512

img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

img_paths.sort()
mask_paths.sort()

os.makedirs(OUTPUT_DIR)
for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE):
        for x in range(0, img.shape[1], TARGET_SIZE):
            img_tile = img[y : y + TARGET_SIZE, x : x + TARGET_SIZE]
            mask_tile = mask[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                out_img_path = os.path.join(
                    OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k)
                )
                cv2.imwrite(out_img_path, img_tile)

                out_mask_path = os.path.join(
                    OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k)
                )
                cv2.imwrite(out_mask_path, mask_tile)

            k += 1

    print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
