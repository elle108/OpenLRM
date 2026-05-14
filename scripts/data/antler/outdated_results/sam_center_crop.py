import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def segment_antler_from_crop(cropped_image_path, output_path):
    print(f"Loading SpeciesNet crop: {cropped_image_path}")
    image = cv2.imread(cropped_image_path)
    if image is None:
        raise ValueError(f"Could not read {cropped_image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Calculate the exact center pixel of the cropped image
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    print(f"Calculated center point: ({center_x}, {center_y})")

    # 2. Load the Segment Anything Model (SAM)
    print("Loading SAM (Segment Anything Model)...")
    sam_checkpoint = "database/models/sam_vit_b_01ec64.pth" # Adjust path if needed
    
    # Force CPU to prevent Mac architecture crashes!
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=torch.device('cpu')) 
    
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # 3. Give SAM the center pixel as a "Point Prompt"
    # The label '1' tells SAM "this point is the foreground object we want to keep"
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])

    print("Prompting SAM to extract the object at the center...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False # We only want the single best mask
    )
    
    # Convert SAM's boolean array into an 8-bit black & white mask
    mask = (masks[0] * 255).astype(np.uint8)

    # 4. Apply the mask to make the background and ear completely transparent
    b, g, r = cv2.split(image)
    rgba = [b, g, r, mask]
    transparent_antler = cv2.merge(rgba)

    # Save the result!
    cv2.imwrite(output_path, transparent_antler)
    print(f"Success! Segmented antler saved to {output_path}")
    
    return output_path

# Quick test block
if __name__ == "__main__":
    # Test it on one of your previously cropped images!
    test_input = "database/extractedAntlers/test_crop.jpg" # Change to an actual filename
    test_output = "database/extractedAntlers/test_segmented.png"
    segment_antler_from_crop(test_input, test_output)