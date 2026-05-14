import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

def process_antlers(image_path, output_prefix):
    print(f"--- STARTING DEEP LEARNING PIPELINE FOR {image_path} ---")
    
    # 1. Load Custom YOLO Model
    print("1. Running Custom YOLO Segmentation...")
    model = YOLO("UPDATE PATH") 
    results = model.predict(image_path, conf=0.25)
    
    # Check if we found at least two antlers
    if len(results[0].masks) < 2:
        print("Warning: Could not find two distinct overlapping antlers.")
        return
        
    # Get the original image dimensions
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    
    # Extract the masks for Antler 1 and Antler 2
    # YOLO normalizes masks, so we resize them back to original image size
    mask1_data = cv2.resize(results[0].masks.data[0].cpu().numpy(), (w, h))
    mask2_data = cv2.resize(results[0].masks.data[1].cpu().numpy(), (w, h))
    
    mask1 = (mask1_data * 255).astype(np.uint8)
    mask2 = (mask2_data * 255).astype(np.uint8)
    
    # Fill any gaps
    kernel = np.ones((15, 15), np.uint8)
    mask1_filled = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask2_filled = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    # 2. Run LaMa Mutual Inpainting
    print("2. Booting LaMa for Deep Reconstruction...")
    lama = SimpleLama()
    pil_image = Image.open(image_path).convert('RGB')
    
    # To isolate Antler 1, erase Antler 2
    print("   -> Erasing Antler 2 to isolate Antler 1...")
    pil_mask2 = Image.fromarray(mask2_filled).convert('L')
    isolated_antler_1 = lama(pil_image, pil_mask2)
    
    # To isolate Antler 2, erase Antler 1
    print("   -> Erasing Antler 1 to isolate Antler 2...")
    pil_mask1 = Image.fromarray(mask1_filled).convert('L')
    isolated_antler_2 = lama(pil_image, pil_mask1)

    # 3. Save Results
    isolated_antler_1.save(f"{output_prefix}_isolated_1.jpg")
    isolated_antler_2.save(f"{output_prefix}_isolated_2.jpg")
    print("--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    process_antlers("Inpaint_Cropped.jpg", "dl_output")