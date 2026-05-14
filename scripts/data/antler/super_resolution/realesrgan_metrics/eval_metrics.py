import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def run_experiment(input_path, model_path, scale):
    print(f"--- Starting Real-ESRGAN Evaluation (Scale: x{scale}) ---")
    
    # Prepare Ground Truth (512x512)
    orig = cv2.imread(input_path)
    if orig is None:
        raise ValueError(f"Could not load image at {input_path}")
    
    gt_img = cv2.resize(orig, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite("GT_512.jpg", gt_img)
    
    # Degrade the image
    new_w, new_h = 512 // scale, 512 // scale
    low_res = cv2.resize(gt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"LowRes_1_{scale}.jpg", low_res)
    print(f"Degraded image to {new_w}x{new_h}")
    
    # Load Real-ESRGAN Model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    
    # Upsample to recover
    print(f"Running Real-ESRGAN recovery to 512x512")
    super_res, _ = upsampler.enhance(low_res, outscale=scale)
    super_res = cv2.resize(super_res, (512, 512)) # Ensure exact dimensions
    cv2.imwrite(f"SuperRes_Recovered_x{scale}.jpg", super_res)
    
    # Calculate Metrics
    print("Calculating PSNR and SSIM")
    p_val = psnr(gt_img, super_res)
    s_val = ssim(gt_img, super_res, multichannel=True, channel_axis=2)
    
    print("\n" + "="*40)
    print(f"RESULTS FOR 1/{scale} RECOVERY:")
    print(f"PSNR: {p_val:.2f} dB")
    print(f"SSIM: {s_val:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--model", required=True, help="Path to the .pth model file")
    ap.add_argument("--scale", type=int, default=4, help="Scale factor (e.g., 4 or 8)")
    args = ap.parse_args()
    
    run_experiment(args.input, args.model, args.scale)