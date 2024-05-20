import os
import json
import numpy as np
from PIL import Image, ImageDraw


def create_mask(image_shape, polygons, labels):
    """
    Create masks for one image.
    
    :param image_shape: the size of the image
    :param polygons: the polygons used for determind where and what mask are created
    :param lables: the labels for the polygons, defines which class the polygon consists of

    :returns: np.array with integers representing the class for each pixel in the image
    """
    # Create an empty mask with background class (0)
    mask = Image.new('L', image_shape, 0)
    draw = ImageDraw.Draw(mask)
    
    # Class dictionary mapping labels to class integers
    class_dict = {'building': 1, 'tree': 2, 'water': 3, 'road': 4}
    
    for polygon, label in zip(polygons, labels):
        if label in class_dict:
            class_value = class_dict[label]
            # Draw the polygon filled with the class-specific value
            draw.polygon(polygon, fill=class_value)
    
    return np.array(mask)


def process_json_files(directory):
    """
    Prosess json files in a directory, and creates masks from the data inside the json files.
    
    :param directory: the directory where the json files are located

    :returns: np.array with masks for all the json files in the directory
    """
    masks = []

    # Iterate over every file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Taking imageWidth and imageHeigth from json file, altho in this project all images are set to 512*512 px.
            image_shape = (data['imageWidth'], data['imageHeight'])
            polygons = []
            labels = []
            
            #Iterates over the different polygons in the jsonfile
            for shape in data['shapes']:
                if 'polygon' in shape['shape_type']:
                    polygon = [(p[0], p[1]) for p in shape['points']]
                    polygons.append(polygon)
                    labels.append(shape['label'])
            
            # Generating mask from the polygons and adding it to the array
            mask = create_mask(image_shape, polygons, labels)
            mask_resized = Image.fromarray(mask).resize((512, 512), Image.NEAREST)
            masks.append(np.array(mask_resized))
    
    return np.array(masks)

# Usage example
# masks_array = process_json_files('ImageExtractor\Images\Classified')
# print(masks_array.shape)
