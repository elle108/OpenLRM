# Colorization

This pipeline converts grayscale/Infrared (IR) nighttime images into colorized images using a deep learning colorization model.

The model analyzes the textures and gradients in the grayscale image (e.g., the texture of a deer vs. the texture of leaves) and applies "hallucinated" colors based on its training data.

Important Note: The generated colors are approximations meant to improve visual clarity and contrast. They should not be used to definitively determine the real world color of an object in the dark.

## Download models

The OpenCV DNN module requires three specific files to run this model: the network architecture (.prototxt), the trained weights (.caffemodel), and the color cluster centers (.npy).

Run the following commands in your terminal to download the required model files directly to your working directory:

```Bash
# 1. Download the network architecture
wget https://huggingface.co/spaces/Kingoteam/colorization_demo/resolve/main/colorization_deploy_v2.prototxt

# 2. Download the trained Caffe weights (123 MB)
wget https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel

# 3. Download the color cluster points
wget https://huggingface.co/spaces/Kingoteam/colorization_demo/resolve/main/pts_in_hull.npy
```

## How to use

This process relies on a Python script to execute the OpenCV model, and a Slurm (.sbatch) script to submit the job to the HPC cluster's GPU partition.

1. Save colorize.py in your project folder
2. Save the Slurm script run_color.sbatch and submit it to the cluster using: ```sbatch run_color.sbatch```

Note: Inside the run_color.sbatch file, you must define the exact absolute paths to your input image, your desired output location, and the model files.
