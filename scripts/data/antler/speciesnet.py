import cv2
from transformers import pipeline
from PIL import Image

def detect_and_crop(input_path: str, output_path: str) -> str:
    """
    Uses zero-shot object detection to find an antler in the image,
    draws a bounding box around it, adds a small padding margin, and crops it.
    """
    try:
        print(f"[SpeciesNet] Scanning {input_path} for antlers")
        
        # 1. Load image for the AI
        image = Image.open(input_path).convert("RGB")
        
        # 2. Initialize the Zero-Shot Object Detector
        detector = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32")
        
        # 3. Ask the AI to find the object. 
        # We lower the threshold to 0.05 (5% confidence) so it doesn't filter out good guesses.
        predictions = detector(
            image,
            candidate_labels=["antler", "horn", "bone", "skull", "object", "head"],
            threshold=0.001 
        )
        
        if not predictions:
            print("[SpeciesNet] No objects met the confidence threshold. Skipping crop.")
            return None

        # Sort predictions by score (highest first) just to be safe
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        
        # Debug: Print the top 3 things it found
        print("AI Top Detections:")
        for i, p in enumerate(predictions[:3]):
            print(f"  {i+1}. {p['label']} ({round(p['score']*100, 2)}%)")

        # 4. Get the best prediction
        best_prediction = predictions[0]
        box = best_prediction["box"]
        print(f"🎯 Selected: {best_prediction['label']} to crop!")

        # 5. Load image with OpenCV for cropping
        img_cv2 = cv2.imread(input_path)
        h, w, _ = img_cv2.shape

        # Extract coordinates
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

        # 6. Add a 5% margin/padding
        margin_x = int((xmax - xmin) * 0.05)
        margin_y = int((ymax - ymin) * 0.05)

        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(w, xmax + margin_x)
        crop_ymax = min(h, ymax + margin_y)

        # 7. Perform the crop
        cropped_img = img_cv2[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # 8. Save the cropped image
        cv2.imwrite(output_path, cropped_img)
        print(f"[SpeciesNet] Crop successful! Saved to {output_path}")
        
        return output_path

    except Exception as e:
        print(f"[SpeciesNet] Error during detection/cropping: {e}")
        return None

if __name__ == "__main__":
    detect_and_crop("TCB2.jpg", "testDeer5_cropped.jpg")