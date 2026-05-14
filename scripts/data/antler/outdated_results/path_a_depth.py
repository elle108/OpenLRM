import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import pipeline
from sklearn.cluster import KMeans

def separate_by_depth(image_path, output_prefix, yolo_model_path="database/models/antler_seg_model.pt"):
    print("1. Running YOLO to find the combined antler mass...")
    yolo = YOLO(yolo_model_path)
    results = yolo.predict(image_path, conf=0.25, retina_masks=True)
    
    if results[0].masks is None:
        print("No antlers found!")
        return
        
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    
    # Combine all YOLO masks into one giant mask
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_data in results[0].masks.data:
        mask_resized = cv2.resize(mask_data.cpu().numpy(), (w, h), interpolation=cv2.INTER_CUBIC)
        yolo_mask = cv2.bitwise_or(yolo_mask, (mask_resized * 255).astype(np.uint8))
        
    print("2. Downloading/Booting Depth AI (Intel DPT)...")
    # This downloads the depth model on the first run
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    
    from PIL import Image
    pil_img = Image.open(image_path).convert('RGB')
    depth_output = depth_estimator(pil_img)
    
    # Convert depth map to numpy array
    depth_map = np.array(depth_output['depth'])
    depth_map = cv2.resize(depth_map, (w, h))
    
    print("3. Slicing 'Front' and 'Back' using 3D Topography...")
    # Get only the depth values that fall INSIDE the YOLO antler mask
    valid_pixels = yolo_mask > 0
    antler_depths = depth_map[valid_pixels].reshape(-1, 1)
    
    # Use AI clustering to group the depth values into 2 distinct layers
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(antler_depths)
    labels = kmeans.labels_
    
    # Figure out which cluster is the "Front" (higher depth value = closer)
    cluster_centers = kmeans.cluster_centers_.flatten()
    front_cluster_idx = np.argmax(cluster_centers)
    
    # Reconstruct the two masks
    mask_front = np.zeros((h, w), dtype=np.uint8)
    mask_back = np.zeros((h, w), dtype=np.uint8)
    
    # Map the labels back to the image coordinates
    mask_front[valid_pixels] = (labels == front_cluster_idx) * 255
    mask_back[valid_pixels] = (labels != front_cluster_idx) * 255
    
    cv2.imwrite(f"{output_prefix}_mask_front.png", mask_front)
    cv2.imwrite(f"{output_prefix}_mask_back.png", mask_back)
    print(f"Done! Check {output_prefix}_mask_front.png and back.png")

if __name__ == "__main__":
    separate_by_depth("Inpaint_Cropped.png", "depth_test")