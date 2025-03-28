import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global index to track which patient is displayed
current_index = 0
patient_ids = []

def load_image(image_path, is_mask=False):
    """Load an image as grayscale."""
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found -> {image_path}")
        return None
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Unable to read image -> {image_path}")
        return None

    if is_mask:
        img = (img > 127).astype(np.uint8)

    return img

def overlay_mask_on_mri(mri, mask, alpha=0.5):
    """Overlay mask on MRI with transparency."""
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color for mask

    mri_colored = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(mri_colored, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

def update_display(fig, axes, mask_dir, mri_dir):
    """Update display for the current patient index."""
    global current_index, patient_ids

    if not patient_ids:
        return
    
    patient_id = patient_ids[current_index]
    pred_mask_img_path = os.path.join(mask_dir, f"P{patient_id}.png")
    mri_img_path = os.path.join(mri_dir, f"P{patient_id}.png")

    mask = load_image(pred_mask_img_path, is_mask=True)
    mri = load_image(mri_img_path)

    if mask is None or mri is None:
        return

    overlay = overlay_mask_on_mri(mri, mask)

    axes[0].imshow(mri, cmap="gray")
    axes[0].set_title(f"MRI - Patient {patient_id}")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Overlay - Patient {patient_id}")
    axes[1].axis("off")

    fig.suptitle(f"Patient {patient_id}", fontsize=14)

    plt.draw()

def on_key(event, fig, axes, mask_dir, mri_dir):
    """Handle key presses for navigation."""
    global current_index, patient_ids

    if event.key == "right":
        current_index = min(current_index + 1, len(patient_ids) - 1)
    elif event.key == "left":
        current_index = max(current_index - 1, 0)

    update_display(fig, axes, mask_dir, mri_dir)

def compare_directories(mask_dir, mri_dir):
    """Compare images in the given directories."""
    global patient_ids, current_index

    # Extract patient IDs from filenames (assuming format "P{number}.png")
    mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.startswith("P") and f.endswith(".png")])
    patient_ids = sorted([int(f[1:-4]) for f in mask_filenames])  # Extract numbers from P{number}.png

    if not patient_ids:
        print(f"❌ No valid patient images found in {mask_dir}")
        sys.exit(1)

    current_index = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    update_display(fig, axes, mask_dir, mri_dir)

    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, fig, axes, mask_dir, mri_dir))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py pred_mask_dir mri_dir")
        sys.exit(1)

    mask_dir = sys.argv[1]
    mri_dir = sys.argv[2]

    if not os.path.isdir(mask_dir) or not os.path.isdir(mri_dir):
        print(f"❌ Error: One of the provided paths is not a directory -> {mask_dir}, {mri_dir}")
        sys.exit(1)

    print(f"📂 Comparing masks from '{mask_dir}' with MRIs from '{mri_dir}'")
    compare_directories(mask_dir, mri_dir)
