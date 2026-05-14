import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from colorize import colorize_image

# ==========================================
# YOUR CUSTOM COLORIZATION FUNCTION HERE
# ==========================================
def run_your_colorizer(grayscale_img):
    # 1. Save the in-memory array to a temporary file on disk
    temp_input = "temp_gray_input.jpg"
    temp_output = "temp_colorized_output.jpg"
    cv2.imwrite(temp_input, grayscale_img)
    
    # 2. Pass the FILE PATHS to your colorizer, not the array!
    # Note: Make sure colorize_image is imported at the top of your script!
    colorize_image(
        temp_input, 
        temp_output, 
        prototxt="database/models/colorization_deploy_v2.prototxt", 
        model="database/models/colorization_release_v2.caffemodel", 
        points="database/models/pts_in_hull.npy"
    )
    
    # 3. Read the newly colorized file back into memory
    colorized_result = cv2.imread(temp_output)
    
    # 4. Clean up the temporary files so they don't clutter your folder
    if os.path.exists(temp_input):
        os.remove(temp_input)
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    return colorized_result

# ==========================================
# THE AUTOMATED MATH PIPELINE
# ==========================================
def calculate_metrics(ground_truth, colorized):
    """Calculates PSNR and SSIM between two images."""
    # Ensure images are exactly the same size before doing math
    colorized = cv2.resize(colorized, (ground_truth.shape[1], ground_truth.shape[0]))
    
    psnr_val = psnr(ground_truth, colorized)
    # channel_axis=2 tells it to look at RGB color channels
    ssim_val = ssim(ground_truth, colorized, channel_axis=2) 
    
    return psnr_val, ssim_val

def run_experiment(dataset_folder, output_csv="colorization_results.csv"):
    print(f"Starting automated colorization test on folder: {dataset_folder}")
    
    # Grab all jpg/png files in the folder
    image_paths = glob(os.path.join(dataset_folder, "*.[jp][pn]g"))
    results = []

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"Processing: {file_name}...")
        
        # 1. Load Ground Truth
        gt_full = cv2.imread(img_path)
        if gt_full is None:
            continue
            
        h, w = gt_full.shape[:2]

        # 2. Simulate the bounding box crop (you can replace this with YOLO later!)
        # For now, we mathematically crop the center 50% of the image
        crop_y1, crop_y2 = int(h*0.25), int(h*0.75)
        crop_x1, crop_x2 = int(w*0.25), int(w*0.75)
        
        gt_crop = gt_full[crop_y1:crop_y2, crop_x1:crop_x2]

        # 3. Convert both to Grayscale (Simulating the inputs)
        gray_full = cv2.cvtColor(gt_full, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2GRAY)

        # 4. Colorize Both Branches
        colorized_full = run_your_colorizer(gray_full)
        colorized_crop = run_your_colorizer(gray_crop)

        # 5. Run the Math!
        full_psnr, full_ssim = calculate_metrics(gt_full, colorized_full)
        crop_psnr, crop_ssim = calculate_metrics(gt_crop, colorized_crop)

        # 6. Save data
        results.append({
            "Image": file_name,
            "Full_PSNR": full_psnr,
            "Full_SSIM": full_ssim,
            "Crop_PSNR": crop_psnr,
            "Crop_SSIM": crop_ssim
        })

    # ==========================================
    # DATA EXPORT & ANALYSIS
    # ==========================================
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*40)
    print("🏆 EXPERIMENT RESULTS 🏆")
    print("="*40)
    print(f"Total Images Tested: {len(df)}")
    print("\nAVERAGE SCORES (Higher is better):")
    print(f"FULL IMAGE - PSNR: {df['Full_PSNR'].mean():.2f} | SSIM: {df['Full_SSIM'].mean():.4f}")
    print(f"CROPPED    - PSNR: {df['Crop_PSNR'].mean():.2f} | SSIM: {df['Crop_SSIM'].mean():.4f}")
    
    # Determine the winner
    if df['Crop_SSIM'].mean() > df['Full_SSIM'].mean():
        print("\n✅ CONCLUSION: Cropping IMPROVES colorization quality.")
    else:
        print("\n❌ CONCLUSION: Cropping HURTS colorization quality. Feed the model full images.")
        
    print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    # Point this to a folder full of your colored ground truth images
    run_experiment("train/")