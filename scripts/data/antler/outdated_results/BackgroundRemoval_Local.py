import cv2
from rembg import remove
from pathlib import Path

def remove_background(input_path, output_path, **kwargs):
    """
    Truly offline background removal using the local rembg library (U2Net).
    Requires no API keys and has no monthly quota.
    (Accepts **kwargs to safely ignore any extra arguments passed by older code).
    """
    print(f"   -> Running offline rembg on {input_path}")
    
    # Read the image using cv2 (cast to string in case a Path object is passed)
    img = cv2.imread(str(input_path))
    
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")
        
    # Process the image locally
    output_img = remove(img)
    
    # Save the output (which will have a transparent 4th alpha channel)
    cv2.imwrite(str(output_path), output_img)
    return str(output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove background using local rembg")
    parser.add_argument("input", help="input image path")
    parser.add_argument("--output", "-o", help="output path (default: <input>NBG.png)")
    args = parser.parse_args()

    infile = args.input
    outfile = args.output or str(Path(infile).with_name(Path(infile).stem + "NBG.png"))
    
    try:
        result = remove_background(infile, output_path=outfile)
        print(f"Background removed locally, saved to: {result}")
    except Exception as e:
        print("Error:", e)