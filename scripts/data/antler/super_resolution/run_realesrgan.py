import cv2
import argparse
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_gan(input_path, output_path, model_path):
    print(f"Loading image from {input_path}...")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {input_path}.")

    print("Loading Real-ESRGAN Architecture...")
    # This is the specific architecture for the RealESRGAN_x4plus model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Initialize the upsampler. We use half-precision (fp16) to make it run faster on your GPU!
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0, # If your GPU runs out of memory, change this to 200 or 400
        tile_pad=10,
        pre_pad=0,
        half=True 
    )

    print("Running Generative Super Resolution (hallucinating textures!)...")
    output, _ = upsampler.enhance(img, outscale=4)
    
    cv2.imwrite(output_path, output)
    print(f"Success! Saved textured high-res image to {output_path}")
    print(f"Original: {img.shape[1]}x{img.shape[0]}  -->  New: {output.shape[1]}x{output.shape[0]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--output", required=True, help="Path to save output image")
    ap.add_argument("--model", default="RealESRGAN_x4plus.pth", help="Path to model weights")
    args = ap.parse_args()

    upscale_gan(args.input, args.output, args.model)