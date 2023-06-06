from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils import filter_segmentation, remove_overlaps
import sys
import cv2
import numpy as np
import base64
import json

sys.path.append("..")

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