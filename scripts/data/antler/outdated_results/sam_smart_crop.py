import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def segment_fence_guard(cropped_image_path, output_path):
    print(f"Loading SpeciesNet crop: {cropped_image_path}")
    image = cv2.imread(cropped_image_path)
    if image is None:
        raise ValueError(f"Could not read {cropped_image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 1. Calculate the Smart Box 
    threshold_value = np.percentile(blurred, 90)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(thresh)
    
    if points is None:
        print("Not enough bright pixels found.")
        return False
        
    x, y, w, h = cv2.boundingRect(points)
    buffer = 20
    h_img, w_img, _ = image.shape
    
    smart_box = np.array([
        max(0, x - buffer), 
        max(0, y - buffer), 
        min(w_img, x + w + buffer), 
        min(h_img, y + h + buffer)
    ])

    # 2. Calculate True Bone Anchor (Positive Point)
    box_mask = np.zeros_like(gray)
    cv2.rectangle(box_mask, (smart_box[0], smart_box[1]), (smart_box[2], smart_box[3]), 255, -1)
    masked_blurred = cv2.bitwise_and(blurred, blurred, mask=box_mask)
    _, _, _, max_loc = cv2.minMaxLoc(masked_blurred)
    
    # 3. Build the Negative Fence (To block the ears)
    neg_points = []
    box_left, box_top, box_right, box_bottom = smart_box
    box_height = box_bottom - box_top
    box_width = box_right - box_left
    
    # We target the bottom 30% of the box, and the outer 25% of the left/right sides
    bottom_y_start = int(box_bottom - (box_height * 0.30))
    left_fence_end = int(box_left + (box_width * 0.25))
    right_fence_start = int(box_right - (box_width * 0.25))

    # Build Left Ear Fence
    for fence_y in range(bottom_y_start, box_bottom, 15): # Drop a point every 15 pixels
        for fence_x in range(box_left, left_fence_end, 15):
            neg_points.append([fence_x, fence_y])

    # Build Right Ear Fence
    for fence_y in range(bottom_y_start, box_bottom, 15):
        for fence_x in range(right_fence_start, box_right, 15):
            neg_points.append([fence_x, fence_y])

    # Combine all points: 1 Positive (Keep), N Negative (Reject)
    point_coords = np.array([[max_loc[0], max_loc[1]]] + neg_points)
    point_labels = np.array([1] + [0] * len(neg_points))

    print(f"Smart Box coordinates: {smart_box}")
    print(f"True Bone Anchor found at: {max_loc}")
    print(f"Deploying {len(neg_points)} negative guard points over ear regions...")

    # 4. Load SAM
    print("Loading SAM (Segment Anything Model)...")
    sam_checkpoint = "database/models/sam_vit_b_01ec64.pth" 
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=torch.device('cpu')) 
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # 5. Prompt SAM with Box + Anchor + Fences
    print("Prompting SAM with Fence Guards active...")
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=smart_box[None, :], 
        multimask_output=False 
    )
    
    mask = (masks[0] * 255).astype(np.uint8)

    # 6. Make background transparent
    b, g, r = cv2.split(image)
    rgba = [b, g, r, mask]
    transparent_antler = cv2.merge(rgba)

    cv2.imwrite(output_path, transparent_antler)
    print(f"Success! Segmented antler saved to {output_path}")
    return True

if __name__ == "__main__":
    test_input = "database/extractedAntlers/test_crop.jpg" 
    test_output = "database/extractedAntlers/test_fence_guard.png"
    segment_fence_guard(test_input, test_output)