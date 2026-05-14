from rembg import remove
from PIL import Image
import requests
import os

# --- API SETTINGS ---
API_URL = "https://api.rembg.com/rmbg"
API_KEY = "122e2659-f5d7-42b8-a238-7e8346c2c182"

# --- STEP 1: Remove Background Via rembg.com API ---
def remove_background_api(input_path):
    # Get image size
    with Image.open(input_path) as img:
        width, height = img.size

    # Input image path
    filename = input_path.split(".")[0]
    #extensions = [".jpg", ".jpeg", ".png", ".webp"]
    output_path = filename + "NBG.png" # Must be PNG File Type for Transparency
    headers = {"x-api-key": API_KEY}
    files = {"image": open(input_path, "rb")}
    data = {
        "format": "png",
        "w": width,
        "h": height,
        "exact_resize": "true",
        "mask": "false",
        "bg_color": "#00000000",
        "angle": 0,
        "expand": "false",
    }

    response = requests.post(API_URL, headers=headers, files=files, data=data)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"API background removal complete → {output_path}") # Notifies User of Completion With API
        return output_path
    else:
        print("API Error:", response.status_code, response.text)
        return None


# --- STEP 2: Local Background Removal Using rembg Library for Transparent Background ---
def remove_background_local(input_path):
    filename = input_path.split(".")[0]
    output_path = filename + "NBGComplete.png"
    # Open and process
    with Image.open(input_path) as img:
        output = remove(img)
    print("Background Removal Complete")
    output.save(output_path)


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    input_path = input("Enter the File Path to the Image: ")

    # Step 1: API Background Removal Results in Black Background
    # Using API First to Handle Complex Backgrounds
    api_output = remove_background_api(input_path)

    # Step 2: Local background removal (can use same input or API result)
    remove_background_local(api_output)
