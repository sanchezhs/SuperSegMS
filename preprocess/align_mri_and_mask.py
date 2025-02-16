import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def align_mri_and_mask(image_dir, mask_dir):
    """
    Applies alignment transformations to all images and masks in the specified directories.

    - Rotates the images 90° clockwise.
    - Flips the masks horizontally.
    """
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            mri = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mri is None or mask is None:
                print(f"Error loading {filename}, skipping...")
                continue

            mri = cv2.rotate(mri, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.flip(mask, 1)

            cv2.imwrite(image_path, mri)
            cv2.imwrite(mask_path, mask)

            print(f"Processed: {filename}")

def overlay_mri_and_mask():
    mri = cv2.imread("datasets/dataset_unet_bias/images/train/P2_T2.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("datasets/dataset_unet_bias/labels/train/P2_T2.png", cv2.IMREAD_GRAYSCALE)
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask > 0] = [0, 255, 0]  # Green
    mri_color = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    overlay = cv2.addWeighted(mri_color, 1.0, colored_mask, alpha, 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay[..., ::-1])  # Convert BGR -> RGB
    plt.title("MRI with Mask Overlay")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    images_paths = [("datasets/dataset_unet_bias/images/train", "datasets/dataset_unet_bias/labels/train"),
                    ("datasets/dataset_unet_bias/images/val", "datasets/dataset_unet_bias/labels/val")]
    for images_dir, masks_dir in images_paths:
        align_mri_and_mask(images_dir, masks_dir)