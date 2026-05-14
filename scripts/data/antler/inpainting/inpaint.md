# Inpainting

This pipeline removes unwanted artifacts, background noise, or impending objects from images using OpenCV's Fast Marching Method (Telea algorithm).

The script analyzes a user provided binary mask (where white pixels indicate the area to be removed) and intelligently samples the healthy pixels immediately surrounding that area to seamlessly blend and reconstruct the missing data.

Important Note: The repaired areas are mathematical approximations based on the surrounding pixels (using a radius of 3). While it works exceptionally well for small blemishes, thin lines, or localized damage, it should not be relied upon to accurately invent large areas of complex, real-world textures.

## Requirements

Unlike the deep learning models, OpenCV's Telea algorithm is built in and does not require you to download any external .caffemodel or .prototxt files.

However, you must provide a mask image alongside your input image.

The mask must be a grayscale/binary image.

White pixels (255) equal the exact area you want to erase.

Black pixels (0) equal the healthy image data to keep.

The mask image must have the exact same pixel dimensions as the input image, or the script will fail.

## How to use

This process relies on a Python script to execute the OpenCV algorithm, and a Slurm (.sbatch) script to submit the job to the HPC cluster's GPU partition.

Save inpaint.py in your project folder.

Ensure your target image and mask image are uploaded to the folder.

Inside the run_inpaint.sbatch file, you must define the exact paths or filenames for your INPUT_IMG, MASK_IMG, and OUTPUT_IMG variables to ensure the script targets the correct files.

Submit run_inpaint.sbatch to the cluster using:

```
sbatch run_inpaint.sbatch
```