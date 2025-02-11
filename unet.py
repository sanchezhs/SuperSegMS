import os
import argparse
import cv2
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Dataset directories
MSLESSEG_TRAIN_DIR = "MSLesSeg-Dataset/train"
DATASET_PATH = "datasets/dataset_unet"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images", "train")
TRAIN_MASK_DIR = os.path.join(DATASET_PATH, "labels", "train")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "images", "val")
VAL_MASK_DIR = os.path.join(DATASET_PATH, "labels", "val")

# Training parameters
BATCH_SIZE = 8
IMG_SIZE = 256  # Resize images
TARGET_SIZE = 256  # Target size for the model
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MRIDataset(Dataset):
    """
    Custom dataset for loading MRI images and corresponding masks.
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image and mask in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize images and masks
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension
        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask

class MRIMultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, patient_dirs, transform=None):
        self.patient_dirs = patient_dirs  # List of directories or file paths
        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        # Load each modality (assuming they are in separate files)
        flair_path = self.patient_dirs[idx]['flair']
        t1_path = self.patient_dirs[idx]['t1']
        t2_path = self.patient_dirs[idx]['t2']
        mask_path = self.patient_dirs[idx]['mask']
        
        # Load the images (example for NIfTI images)
        flair_img = nib.load(flair_path).get_fdata()
        t1_img = nib.load(t1_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()
        
        # Select the best slice (you might need a more robust method)
        best_slice_idx = np.argmax(np.sum(flair_img, axis=(0, 1)))
        flair_slice = cv2.resize(flair_img[:, :, best_slice_idx], (TARGET_SIZE, TARGET_SIZE))
        t1_slice = cv2.resize(t1_img[:, :, best_slice_idx], (TARGET_SIZE, TARGET_SIZE))
        t2_slice = cv2.resize(t2_img[:, :, best_slice_idx], (TARGET_SIZE, TARGET_SIZE))
        mask_slice = cv2.resize(mask_img[:, :, best_slice_idx], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Normalize each modality to [0, 1]
        epsilon = 1e-8 # Small value to avoid division by zero
        flair_slice = flair_slice.astype(np.float32) / (flair_slice.max() + epsilon)
        t1_slice = t1_slice.astype(np.float32) / (t1_slice.max() + epsilon)
        t2_slice = t2_slice.astype(np.float32) / (t2_slice.max() + epsilon)
        mask_slice = mask_slice.astype(np.float32)  # Assuming mask is already binary
        
        # Stack the modalities to form a 3-channel image
        image = np.stack([flair_slice, t1_slice, t2_slice], axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

def train_model(multimodal=False):
    """
    Train the U-Net model using either single-modality or multimodal inputs.
    
    If multimodal is False, the model will use a single channel (grayscale) and load data from TRAIN_IMG_DIR/ TRAIN_MASK_DIR.
    If multimodal is True, the model will use three channels (FLAIR, T1-w, T2-w) and generate the sample dictionaries from the MSLesSeg directory.
    """
    if multimodal:
        samples = generate_multimodal_samples(MSLESSEG_TRAIN_DIR)
        train_dataset = MRIMultiModalDataset(samples)
        in_channels = 3
    else:
        train_dataset = MRIDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
        in_channels = 1

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = smp.Unet(encoder_name="resnet34", in_channels=in_channels, classes=1, activation=None)
    model.to(DEVICE)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Save the trained model
    if multimodal:
        torch.save(model.state_dict(), "unet_multimodal_model.pth")
        print("Multimodal U-Net model saved successfully.")
    else:
        torch.save(model.state_dict(), "unet_model.pth")
        print("U-Net model saved successfully.")


def generate_multimodal_samples(train_dir):
    """
    Generate a list of dictionaries for multimodal samples from the MSLesSeg training directory.
    Expected structure:
      - train_dir/
          Patient_ID/
              Timepoint/
                  PatientID_Timepoint_FLAIR.nii.gz
                  PatientID_Timepoint_T1.nii.gz
                  PatientID_Timepoint_T2.nii.gz
                  PatientID_Timepoint_MASK.nii.gz
    """
    samples = []
    for patient in os.listdir(train_dir):
        patient_path = os.path.join(train_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        for timepoint in os.listdir(patient_path):
            timepoint_path = os.path.join(patient_path, timepoint)
            if not os.path.isdir(timepoint_path):
                continue
            flair_path = os.path.join(timepoint_path, f"{patient}_{timepoint}_FLAIR.nii.gz")
            t1_path = os.path.join(timepoint_path, f"{patient}_{timepoint}_T1.nii.gz")
            t2_path = os.path.join(timepoint_path, f"{patient}_{timepoint}_T2.nii.gz")
            mask_path = os.path.join(timepoint_path, f"{patient}_{timepoint}_MASK.nii.gz")
            if os.path.exists(flair_path) and os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(mask_path):
                samples.append({
                    'flair': flair_path,
                    't1': t1_path,
                    't2': t2_path,
                    'mask': mask_path
                })
    return samples



def compute_metrics(pred, mask):
    """
    Compute Intersection over Union (IoU) and Dice Score.

    Both pred and mask should be in the range [0, 1]. Threshold at 0.5 to create binary masks.
    """
    # Threshold predictions and masks at 0.5
    pred_bin = (pred > 0.5).astype(np.uint8)
    mask_bin = (mask > 0.5).astype(np.uint8)

    intersection = np.logical_and(pred_bin, mask_bin).sum()
    union = np.logical_or(pred_bin, mask_bin).sum()
    iou = intersection / union if union > 0 else 0

    dice = (
        (2.0 * intersection) / (pred_bin.sum() + mask_bin.sum())
        if (pred_bin.sum() + mask_bin.sum()) > 0
        else 0
    )
    return iou, dice


def evaluate_model(model_path, multimodal=False):
    """
    Evaluate the U-Net model on the validation dataset and compute additional metrics including a confusion matrix.
    
    If multimodal is True, the function loads a model with in_channels=3 and reads images as 3-channel color images.
    Otherwise, it uses in_channels=1 and reads grayscale images.
    """
    # Initialize model with the appropriate number of input channels
    if multimodal:
        model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
    else:
        model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1, activation=None)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    ious, dices = [], []
    sample_images = []  # To store a few samples for display
    total_cm = np.zeros((2, 2), dtype=np.int64)

    # Iterate over validation images
    for filename in sorted(os.listdir(VAL_IMG_DIR)):
        img_path = os.path.join(VAL_IMG_DIR, filename)
        mask_path = os.path.join(VAL_MASK_DIR, filename)

        # Load ground truth mask in grayscale and normalize to [0,1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = mask  # Assuming images are already at the target size
        mask_resized = mask_resized.astype(np.float32) / 255.0

        if multimodal:
            # Read multi-channel image (assumed to be saved via early fusion)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Shape: (H, W, 3) in BGR order
            # Convert BGR to RGB (if desired)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            # Convert from (H, W, 3) to (3, H, W)
            img = np.transpose(img, (2, 0, 1))
        else:
            # Read grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            # Expand dims to have shape (1, H, W)
            img = np.expand_dims(img, axis=0)

        # Prepare the tensor: add batch dimension
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Get model prediction
        with torch.no_grad():
            pred_mask = model(img_tensor)
            pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()

        # Compute IoU and Dice Score
        iou, dice = compute_metrics(pred_mask, mask_resized)
        ious.append(iou)
        dices.append(dice)

        # Compute confusion matrix for this image
        pred_bin = (pred_mask > 0.5).astype(np.uint8)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)
        cm = confusion_matrix(mask_bin.flatten(), pred_bin.flatten(), labels=[0, 1])
        total_cm += cm

        # Save sample images (up to 3 samples)
        if len(sample_images) < 3:
            # For display, convert the input image back to a displayable format:
            if multimodal:
                disp_img = np.transpose(img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            else:
                disp_img = img[0]
            sample_images.append((disp_img, (mask_resized * 255).astype(np.uint8), pred_mask))

    # Calculate aggregate metrics from the confusion matrix
    TN, FP, FN, TP = total_cm.ravel()
    accuracy = (TP + TN) / total_cm.sum() if total_cm.sum() > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Display overall metrics
    print(f"Average IoU: {np.mean(ious):.4f}")
    print(f"Average Dice Score: {np.mean(dices):.4f}")
    print("\nConfusion Matrix (Pixels):")
    print(total_cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Display segmentation examples
    plt.figure(figsize=(12, 8))
    for i, (img_disp, mask_disp, pred) in enumerate(sample_images):
        plt.subplot(3, 3, i * 3 + 1)
        if multimodal:
            plt.imshow(img_disp)  # RGB image
        else:
            plt.imshow(img_disp, cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(mask_disp, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(pred, cmap="gray")
        plt.title("U-Net Prediction")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate U-Net model")
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Action to perform: 'train' to train the model or 'evaluate' to evaluate the model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unet_model.pth",
        help="Path to the saved model for evaluation.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use multimodal inputs from the MSLesSeg dataset.",
    )
    args = parser.parse_args()

    if args.action == "train":
        train_model(args.multimodal)
    elif args.action == "evaluate":
        evaluate_model(args.model_path, args.multimodal)
    else:
        raise ValueError("Invalid action. Use 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()

# Average Dice Score: 0.5085