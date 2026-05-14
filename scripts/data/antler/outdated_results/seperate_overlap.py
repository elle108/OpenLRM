import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

def fill_and_expand_mask(mask_array, fill_kernel=25, expand_kernel=15):
    """
    Bridges gaps in the mask, then slightly expands (dilates) it.
    Expanding is crucial for LaMa so it doesn't leave thin 'ghost' edges of the erased antler.
    """
    # 1. Close the gaps
    kernel_close = np.ones((fill_kernel, fill_kernel), np.uint8)
    closed_mask = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Expand the mask slightly to cover edges
    kernel_dilate = np.ones((expand_kernel, expand_kernel), np.uint8)
    expanded_mask = cv2.dilate(closed_mask, kernel_dilate, iterations=1)
    
    return expanded_mask

def separate_antlers(image_path, output_prefix, model_path="database/models/antler_seg_model.pt"):
    print(f"--- STARTING OVERLAP SEPARATION FOR {image_path} ---")
    
    print("1. Running Custom YOLO Segmentation...")
    model = YOLO(model_path) 
    results = model.predict(image_path, conf=0.25, retina_masks=True)
    
    if results[0].masks is None or len(results[0].masks) < 2:
        print("Error: Could not find at least two distinct antlers.")
        return
        
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    
    # --- THE FIX: Sort masks by physical size (area) ---
    print("2. Isolating the two largest antler structures...")
    masks_np = results[0].masks.data.cpu().numpy()
    
    # Calculate the sum of pixels for each mask to find the area
    areas = [np.sum(m) for m in masks_np]
    
    # Get the indices of the two largest masks
    largest_indices = np.argsort(areas)[-2:] 
    
    # Extract the two largest (index 1 is the absolute largest, index 0 is second largest)
    mask1_data = cv2.resize(masks_np[largest_indices[1]], (w, h), interpolation=cv2.INTER_CUBIC)
    mask2_data = cv2.resize(masks_np[largest_indices[0]], (w, h), interpolation=cv2.INTER_CUBIC)
    
    mask1 = (mask1_data * 255).astype(np.uint8)
    mask2 = (mask2_data * 255).astype(np.uint8)
    
    # 3. Automatically fill gaps and expand edges
    print("3. Bridging gaps and preparing LaMa masks...")
    mask1_ready = fill_and_expand_mask(mask1)
    mask2_ready = fill_and_expand_mask(mask2)

    cv2.imwrite(f"{output_prefix}_mask1_ready.png", mask1_ready)
    cv2.imwrite(f"{output_prefix}_mask2_ready.png", mask2_ready)

    # 4. Deep Learning Inpainting
    print("4. Booting LaMa for Deep Reconstruction (This may take a moment)...")
    lama = SimpleLama()
    pil_image = Image.open(image_path).convert('RGB')
    
    print("   -> Erasing Antler 2 to isolate Antler 1...")
    pil_mask2 = Image.fromarray(mask2_ready).convert('L')
    isolated_antler_1 = lama(pil_image, pil_mask2)
    
    print("   -> Erasing Antler 1 to isolate Antler 2...")
    pil_mask1 = Image.fromarray(mask1_ready).convert('L')
    isolated_antler_2 = lama(pil_image, pil_mask1)

    isolated_antler_1.save(f"{output_prefix}_isolated_1.jpg")
    isolated_antler_2.save(f"{output_prefix}_isolated_2.jpg")
    print("\n--- SEPARATION COMPLETE ---")
    print(f"Check your folder for the '{output_prefix}' files!")

if __name__ == "__main__":
    input_image = "Inpaint_Cropped.png"
    output_name = "overlap_test_v2"
    
    separate_antlers(input_image, output_name)