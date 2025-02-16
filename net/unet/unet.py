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

custom_decoder_channels = [512, 256, 128, 64, 32]

# Create a U-Net model with a resnest34 backbone
# model = smp.Unet(
#     encoder_name="timm-resnest50d",      # Use ResNeSt34 as the backbone
#     encoder_weights="imagenet",          # Set to "imagenet" if pre-trained weights are available and desired
#     in_channels=1,                 # Number of input channels (e.g., 1 for grayscale)
#     classes=1,                     # Number of output classes
#     decoder_channels=custom_decoder_channels  # Custom filters for the decoder layers
# )

# Training parameters
BATCH_SIZE = 10
EPOCHS = 75
LR = 0.0005
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

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension
        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask

def setup_dirs(input_path: str, output_path: str, scale: int):
    """
    Setup global directories based on the input dataset directory and output directory.
    
    The expected dataset structure is:
        input_path/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                
    The output directory will be used to save results and a subfolder 'models' will be used
    to store the trained models.
    """
    global TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, OUTPUT_RESULTS, MODEL_DIR, MSLESSEG_TRAIN_DIR

    # Setup dataset directories
    TRAIN_IMG_DIR = os.path.join(input_path, "images", "train")
    TRAIN_MASK_DIR = os.path.join(input_path, "labels", "train")
    VAL_IMG_DIR = os.path.join(input_path, "images", "val")
    VAL_MASK_DIR = os.path.join(input_path, "labels", "val")
    TEST_IMG_DIR = os.path.join(input_path, "images", "test")
    
    # Setup output directories
    OUTPUT_RESULTS = os.path.join("unet", output_path)
    os.makedirs(OUTPUT_RESULTS, exist_ok=True)
    
    # Create a subdirectory to save/load models within the output directory
    MODEL_DIR = os.path.join(OUTPUT_RESULTS, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # MSLesSeg dataset directory (left unchanged)
    MSLESSEG_TRAIN_DIR = "MSLesSeg-Dataset/train"

def load_model(model_path):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def train_model(scale: int):
    """
    Train the U-Net model
    """
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
    model_name = f"unet_model_sr_x{scale}.pth" if scale > 1 else "unet_model.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_path)
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


def evaluate_model(model_path):
    """
    Evaluate the U-Net model on the test dataset and compute additional metrics including a confusion matrix.
    """
    # Initialize model with the appropriate number of input channels
    model = load_model(model_path)

    ious, dices = [], []
    total_cm = np.zeros((2, 2), dtype=np.int64)

    def add_subtitle(img, text):
        """Helper function to add subtitle to an image"""
        # Create a white padding at the bottom for text
        padding = 30  # Height of the padding for text
        h, w = img.shape
        padded_img = np.zeros((h + padding, w), dtype=np.uint8) + 255
        padded_img[:h, :] = img
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2  # Center text horizontally
        text_y = h + 20  # Position text in padding
        cv2.putText(padded_img, text, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        
        return padded_img

    # Iterate over validation images
    for filename in sorted(os.listdir(VAL_IMG_DIR)):
        img_path = os.path.join(VAL_IMG_DIR, filename)
        mask_path = os.path.join(VAL_MASK_DIR, filename)

        # Load ground truth mask in grayscale and normalize to [0,1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = mask  # Assuming images are already at the target size
        mask_resized = mask_resized.astype(np.float32) / 255.0

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

        # Save predicted images
        input_img = (img[0] * 255).astype(np.uint8)
        ground_truth_img = (mask_resized * 255).astype(np.uint8)
        pred_img = ((pred_mask > 0.5) * 255).astype(np.uint8) # Remove > 0.5 for a smother image

        input_with_subtitle = add_subtitle(input_img, "MRI Brain Image")
        ground_truth_with_subtitle = add_subtitle(ground_truth_img, "Ground Truth")
        pred_with_subtitle = add_subtitle(pred_img, "Prediction")
        
        # Combine images horizontally
        combined_img = np.concatenate([
            input_with_subtitle, 
            ground_truth_with_subtitle, 
            pred_with_subtitle
        ], axis=1)
        
        result_dir = os.path.join(OUTPUT_RESULTS, filename.replace(".png", ""))
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, filename)
        cv2.imwrite(output_path, combined_img)

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

def predict_test(model_path):
    model = load_model(model_path)
    
    for filename in sorted(os.listdir(TEST_IMG_DIR)):
        img_path = os.path.join(TEST_IMG_DIR, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_mask = model(img_tensor)
            pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
        
        # Guardar predicciones
        result_dir = os.path.join(OUTPUT_RESULTS, "test", filename.replace(".png", ""))
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(os.path.join(result_dir, "input.png"), (img[0] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(result_dir, "prediction.png"), (pred_mask * 255).astype(np.uint8))
    
    print("Predictions saved on 'results_unet/test/'.")

def main2():
    parser = argparse.ArgumentParser(description="Train or evaluate U-Net model")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input image or directory containing images for prediction.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results_unet",
        help="Path to save the output predictions",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="Action to perform: 'train' to train the model, 'evaluate' to evaluate the model or 'predict'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unet_model.pth",
        help="Path to the saved model for evaluation.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Super resolution scale factor",
        choices=[1, 2, 3, 4]
    )

    args = parser.parse_args()

    setup_dirs(args.input_path, args.output_path, args.scale)

    if args.action == "train":
        train_model(args.scale)
    elif args.action == "evaluate":
        model_parts = args.model_path.split(".")

        if args.scale > 1:
            model_parts[-2] += f"_sr_x{args.scale}"
            model_path = ".".join(model_parts)
        else:
            model_path = args.model_path
        model_path = os.path.join("unet", "models", model_path)
        evaluate_model(model_path)
    elif args.action == "predict":
        predict_test(args.model_path)
    else:
        raise ValueError("Invalid action. Use 'train' or 'evaluate'.")

def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or predict using the U-Net model"
    )
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train the U-Net model")
    train_parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input dataset directory (required for training)."
    )
    train_parser.add_argument(
        "--output_path",
        type=str,
        default="results_unet",
        help="Path to save training outputs and models (default: results_unet)."
    )
    train_parser.add_argument(
        "--scale",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Super resolution scale factor (default: 1)."
    )

    # Subparser for evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the U-Net model")
    eval_parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to read images from."
    )
    eval_parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save evaluation results."
    )
    eval_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model for evaluation."
    )

    # Subparser for prediction (test)
    predict_parser = subparsers.add_parser("predict", help="Predict using the U-Net model on test images")
    # eval_parser.add_argument(
    #     "--input_path",
    #     type=str,
    #     required=True,
    #     help="Path to read images from."
    # )
    predict_parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save prediction results."
    )
    predict_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model for prediction."
    )

    args = parser.parse_args()

    if args.action == "train":
        setup_dirs(args.input_path, args.output_path, args.scale)
        train_model(args.scale)
    elif args.action == "evaluate":
        setup_dirs(args.input_path, args.output_path, 1)
        # model_path = os.path.join("unet", "models", args.model_path)
        evaluate_model(args.model_path)
    elif args.action == "predict":
        setup_dirs(args.input_path, args.output_path, 1)
        model_path = os.path.join("unet", "models", args.model_path)
        predict_test(model_path)
    else:
        raise ValueError("Invalid action. Use 'train', 'evaluate', or 'predict'.")



if __name__ == "__main__":
    main()