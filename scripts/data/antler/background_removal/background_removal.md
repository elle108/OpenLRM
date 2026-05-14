# Background Removal

This pipeline automatically removes the background from antler images to create clean, transparent assets for 3D model training.

The script uses an AI tool (rembg) to cut out the subject. To make the AI more accurate, it pairs it with 3D spatial data (COLMAP) to figure out exactly where the antler is in physical space, ensuring it ignores background clutter like chairs, tables, or whiteboards. It also features an automatic outlier detection system that flags images where the background removal likely failed.

Important Note: The AI relies on visual contrast and struggles to separate objects that are physically touching. If a person is holding the antler, the AI will often treat the human and the antler as one single object. You will need to manually review the generated outlier report to catch and remove these specific errors.

## Requirements

Unlike the simpler image enhancement scripts, this pipeline requires pre-calculated 3D camera data to correctly guide the AI.

For each antler, your working directory must contain:

* The raw target images.
* Completed COLMAP data (cameras.txt, images.txt, and points3D.txt).

## How to use

This process relies on a Python script to execute the background removal, and a Slurm (.sbatch) script to submit the heavy processing job to the HPC cluster.

Save the Python script in your project folder.

Ensure your raw images and COLMAP data are properly structured in your working directory.

Save your Slurm script and submit it to the cluster using:

```
sbatch run_bg_remove.sbatch
```

Inside your .sbatch file, you must define the following flags for the Python script:

--working_dir: The base directory where your raw antler folders and COLMAP data live.

--view_dir: The output directory where you want the transparent images and JSON split files saved.

--antler_ids: The specific antlers you want to process (e.g., antler_6 antler_7).

## Reviewing Results

Because the AI isn't perfect, the script automatically generates a text file called outlier_report.txt in your output directory.

Always check this file after a job finishes. It will list the specific image numbers (e.g., Image 042) that have an unusually large amount of pixels left over, which usually indicates that a person or large object was accidentally left in the background.