from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
import cv2
import numpy as np
import base64
import json

sys.path.append("..")


def filter_segmentation(masks, lower_size_threshold, upper_size_threshold, confidence_threshold=0.9):
    """Filter segmentation masks based on size and confidence.

    Args:
        masks [dict[area, predicted_iou, bbox]]: Segmentation
        lower_size_threshold (float): Lower size threshold
        upper_size_threshold (float): Upper size threshold
        confidence_threshold (float): Confidence threshold
    """
    filtered_masks = []
    for mask in masks:
        area = mask['area']
        confidence = mask['predicted_iou']
        if lower_size_threshold < area < upper_size_threshold and confidence > confidence_threshold:
            filtered_masks.append(mask)
    return filtered_masks

def remove_overlaps(masks, intersection_threshold=0.5):
    """
    Remove overlapping bounding boxes.
    Args:
        masks (dict[area, confidence, bbox]): Segmentation
        intersection_threshold (float): Intersection threshold
    """
    # Iterate over a copy of the list of dictionaries
    for i, dict1 in enumerate(masks.copy()):
        # Iterate over all the other dictionaries in the copy of the list
        for j, dict2 in enumerate(masks[i+1:].copy(), start=i+1):
            bbox1 = dict1['bbox']
            bbox2 = dict2['bbox']

            # Calculate the area of overlap between the two bounding boxes
            x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
            y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
            overlap_area = x_overlap * y_overlap

            # Calculate the area of each bounding box
            bbox1_area = dict1['area']
            bbox2_area = dict2['area']

            # Calculate the percentage of overlap
            overlap_percentage = overlap_area / min(bbox1_area, bbox2_area)
            # Remove the bounding box with the higher score if they overlap by more than 30%
            if overlap_percentage > intersection_threshold:
                if dict1['area'] < dict2['area']:
                    try:
                        masks.remove(dict1)
                        break
                    except:
                        pass
                else:
                    try:
                        masks.remove(dict2)
                    except:
                        pass

    return masks

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
    
    def predict(
        self,
        image: str = Input(description="Base64 encoded input image"),
        points_per_side: int = Input(default = 20, description= "The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, point_grids must provide explicit point sampling.")
        ) -> str:
        """Run a single prediction on the model"""

        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_batch = 128,
            points_per_side = points_per_side,
        )
        
        # convert the base64 string to bytes
        image_data = base64.b64decode(image)
        # convert the bytes to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # decode the numpy array as an image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the image
        height, width = image.shape[:2]
        # Define the new width and height
        new_width = 512
        new_height = int(new_width * (height / width))
        # Resize the image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        masks = mask_generator.generate(image)

        # filter masks
        image_area = image.shape[0] * image.shape[1]
        lower_area = image_area * (0.05 ** 2)
        upper_area = image_area * (0.8 ** 2)
        masks = filter_segmentation(masks, lower_area, upper_area)
        masks = remove_overlaps(masks, 0.01)
        print("Filtered masks to", len(masks))
        
        # convert masks to json
        json_masks = []
        for mask in masks:
            json_masks.append(mask['bbox'])

        # map the bounding boxes back to the original image size
        for json_mask in json_masks:
            json_mask[0] = int(json_mask[0] * (width / new_width))
            json_mask[1] = int(json_mask[1] * (height / new_height))
            json_mask[2] = int(json_mask[2] * (width / new_width))
            json_mask[3] = int(json_mask[3] * (height / new_height))

        return json.dumps(json_masks)