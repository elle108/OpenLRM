import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def segment_antler_yolo_sam(image_path, output_path):
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. YOLO-World: Find the Antler Bounding Box
    print("Loading YOLO-World...")
    # This automatically downloads the lightweight YOLO-World model locally
    yolo_model = YOLO("yolov8s-world.pt") 
    
    # Define our custom zero-shot class
    yolo_model.set_classes(["antlers", "animal horns", "deer rack", "bone"])
    
    print("Searching for antlers...")
    # conf=0.1 ensures it aggressively looks for antlers even in blurry/dark photos
    results = yolo_model.predict(image_rgb, conf=0.1) 
    
    if len(results[0].boxes) == 0:
        print("No antlers found by YOLO-World!")
        return False
        
    # Extract the coordinates of the highest-confidence bounding box [x_min, y_min, x_max, y_max]
    best_box = results[0].boxes[0].xyxy[0].cpu().numpy()
    print(f"Found antler bounding box: {best_box}")

    # 2. SAM: Cut it out using the Bounding Box
    print("Loading SAM...")
    sam_checkpoint = "database/models/sam_vit_b_01ec64.pth" # Adjust path if needed
    
    # Force CPU to prevent Mac architecture crashes
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=torch.device('cpu')) 
    
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # 3. Prompt SAM with the YOLO bounding box
    input_box = np.array(best_box)
    
    print("Prompting SAM to extract the object inside the bounding box...")
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :], 
        multimask_output=True # Changed to True!
    )
    
    # Calculate the area (number of white pixels) of each of the 3 masks
    areas = [mask.sum() for mask in masks]
    
    # Find the index of the mask with the smallest area
    smallest_mask_idx = np.argmin(areas)
    print(f"SAM generated 3 masks. Selecting the most specific one (Index {smallest_mask_idx}).")
    
    # Convert that specific mask into an 8-bit image
    mask = (masks[smallest_mask_idx] * 255).astype(np.uint8)

    # 4. Apply the mask to make the background transparent
    b, g, r = cv2.split(image)
    rgba = [b, g, r, mask]
    transparent_antler = cv2.merge(rgba)

    # Save the result
    cv2.imwrite(output_path, transparent_antler)
    print(f"Success! Segmented antler saved to {output_path}")
    
    return True

# Quick test block
if __name__ == "__main__":
    # Test it on the exact same image that failed previously!
    test_input = "testDeer.jpg" # Change to your actual file
    test_output = "test_segmented_yolo.png"
    segment_antler_yolo_sam(test_input, test_output)