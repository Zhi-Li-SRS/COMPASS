import glob
import os

import numpy as np
import tifffile


def process_images():
    hsi_path = os.path.join("HSI_data", "1-Wt_FB.tif")
    hsi_img = tifffile.imread(hsi_path)  # e.g., (N, H, W)

    mask_pattern = os.path.join("predicted_results", "hsi_5sub_prediction", "wt_fb", "*.tif")
    mask_files = glob.glob(mask_pattern)

    # Create output directory
    out_dir = "post_processed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for mask_file in mask_files:
        mask = tifffile.imread(mask_file)
        if hsi_img.ndim == 3 and mask.ndim == 2:
            mask = mask[None, ...]  # Add channel dimension
        # Make sure mask is a binary image
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask_norm = mask / 255.0
        result = hsi_img * mask_norm
        # Compose output filename based on mask filename
        base_name = os.path.splitext(os.path.basename(mask_file))[0]
        out_file = os.path.join(out_dir, f"{base_name}_hsi.tif")
        tifffile.imwrite(out_file, result.astype(hsi_img.dtype))


if __name__ == "__main__":
    process_images()
