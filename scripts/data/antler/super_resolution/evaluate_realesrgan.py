import cv2
import matplotlib.pyplot as plt

def create_comparison(orig_path, gan_path, save_path):
    # Load the images
    img_orig = cv2.imread(orig_path)
    img_gan = cv2.imread(gan_path)

    # Convert BGR to RGB for accurate matplotlib colors
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_gan = cv2.cvtColor(img_gan, cv2.COLOR_BGR2RGB)

    # Resize original to match GAN output using basic Bicubic interpolation
    height, width = img_gan.shape[:2]
    img_bicubic = cv2.resize(img_orig, (width, height), interpolation=cv2.INTER_CUBIC)

    # Define a zoomed-in crop area
    crop_size = 200
    start_y, start_x = height // 2 - crop_size // 2, width // 2 - crop_size // 2
    
    crop_bicubic = img_bicubic[start_y:start_y+crop_size, start_x:start_x+crop_size]
    crop_gan = img_gan[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bicubic Upscaling vs. Real-ESRGAN Textures', fontsize=16)

    # Plot Full Images
    axes[0, 0].imshow(img_bicubic)
    axes[0, 0].set_title('Standard Bicubic Upscale')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_gan)
    axes[0, 1].set_title('Real-ESRGAN Upscale')
    axes[0, 1].axis('off')

    # Plot Cropped Patches
    axes[1, 0].imshow(crop_bicubic)
    axes[1, 0].set_title('Bicubic (Zoomed)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(crop_gan)
    axes[1, 1].set_title('Real-ESRGAN (Zoomed)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison to {save_path}")

if __name__ == '__main__':
    original_file = '/mnt/home/milleel7/antler_project/Antler-Super-resolution.jpg'
    gan_file = '/mnt/home/milleel7/antler_project/High_Res_GAN_Antler.jpg'
    output_file = '/mnt/home/milleel7/antler_project/Comparison_Grid.png'
    
    create_comparison(original_file, gan_file, output_file)