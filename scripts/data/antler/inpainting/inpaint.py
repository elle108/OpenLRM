import cv2
import argparse

def run_inpainting(input_path, mask_path, output_path):
    # Load original image
    img = cv2.imread(input_path)
    
    # Load mask as grayscale (white = area to remove, black = keep)
    mask = cv2.imread(mask_path, 0) 
    
    if img.shape[:2] != mask.shape[:2]:
        print("Error: Image and Mask dimensions must match perfectly!")
        return

    print("Running Telea Inpainting algorithm...")
    # 3 is the inpaint radius (how far it looks around the mask to pull pixels from)
    inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, inpainted_img)
    print(f"Saved inpainted image to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    run_inpainting(args.input, args.mask, args.output)