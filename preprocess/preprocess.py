import os
import datetime
import numpy as np
import torch
import nibabel as nib
import cv2
import random

from PIL import Image
from torchvision import transforms

from net.FSRCNN.fsrcnn import FSRCNN


class PreprocessingStep:
    """
    Abstract base class for all preprocessing steps.
    Each subclass must implement the apply method.
    """

    def apply(self, item: dict) -> dict:
        """
        Process the input item and return it.

        :param item: Dictionary containing image data and metadata.
        :return: Updated item.
        """
        raise NotImplementedError("Each preprocessing step must implement the apply method.")


class NIfTItoPNG(PreprocessingStep):
    """
    A preprocessing step for converting a NIfTI volume to PNG slices.
    This step saves PNG slices to a specified output directory and then
    updates the item so that its "data" field becomes a list of 2D images
    loaded from the saved PNG files.
    """

    def __init__(self, output_dir: str, normalize: bool = True):
        """
        Initialize the converter.

        :param output_dir: Directory where PNG images will be saved.
        :param normalize: Whether to normalize pixel values (0-255).
        """
        self.output_dir = output_dir
        self.normalize = normalize
        os.makedirs(self.output_dir, exist_ok=True)

    def apply(self, item: dict) -> dict:
        """
        Convert the NIfTI volume (in item["data"]) to multiple PNG slices.
        The item is updated so that "data" becomes a list of 2D numpy arrays
        (one per slice).

        :param item: Dictionary containing:
                     - "data": a nibabel Nifti1Image or its loaded volume
                     - "file_path": the source file path
        :return: The updated item.
        """
        # If the input data is a nibabel image, get the volume; otherwise, assume it's already a volume.
        if hasattr(item["data"], "get_fdata"):
            image_data = item["data"].get_fdata()
        else:
            image_data = item["data"]

        # Normalize pixel values to 0-255 if required.
        if self.normalize:
            image_data = self.__normalize(image_data)

        # Get a base filename from the file path.
        filename = os.path.basename(item["file_path"]).replace(".nii.gz", "").replace(".nii", "")
        
        # Save PNG slices and get the list of PNG file paths.
        # png_paths = self.__save_slices(image_data, filename)
        best_slice_path = self.__save_best_slices(image_data, filename)
        png_paths = [best_slice_path]
        
        # Load each PNG image (in grayscale) and update the item data.
        slices = []
        for png_path in png_paths:
            slice_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if slice_img is None:
                raise ValueError(f"Failed to load saved PNG image: {png_path}")
            slices.append(slice_img)
        
        # Replace the data with the list of slices.
        item["data"] = slices
        return item

    def __normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image pixel values to the range [0, 255].

        :param image: Numpy array of the image.
        :return: Normalized image array.
        """
        image = image - np.min(image)  # Shift minimum to 0
        if np.max(image) > 0:
            image = (image / np.max(image)) * 255.0  # Scale to [0, 255]
        return image.astype(np.uint8)

    def __save_best_slices(self, image: np.ndarray, filename: str) -> str:
        """
        Save the best slice of the 3D volume as a PNG file and return the list of file paths.

        :param image: 3D numpy array.
        :param filename: Base filename for saving slices.
        :return: List of file paths for the saved PNG slices.
        """
        num_slices = image.shape[-1]
        best = None

        for i in range(num_slices):
            slice_img = image[:, :, i]
            if best is None or np.sum(slice_img) > np.sum(best):
                best = slice_img
        
        slice_filename = os.path.join(self.output_dir, f"{filename}_best.png")
        cv2.imwrite(slice_filename, best)
        return slice_filename


    def __save_slices(self, image: np.ndarray, filename: str) -> list:
        """
        Save each slice of the 3D volume as a PNG file and return the list of file paths.

        :param image: 3D numpy array.
        :param filename: Base filename for saving slices.
        :return: List of file paths for the saved PNG slices.
        """
        png_paths = []
        # Assume the last axis represents the slice dimension.
        num_slices = image.shape[-1]
        for i in range(num_slices):
            slice_img = image[:, :, i]
            slice_filename = os.path.join(self.output_dir, f"{filename}_slice_{i:03d}.png")
            cv2.imwrite(slice_filename, slice_img)
            png_paths.append(slice_filename)

        print(f"Saved {num_slices} slices for {filename} in {self.output_dir}")
        return png_paths


class FsrcnnStep(PreprocessingStep):
    """
    A preprocessing step for applying the FSRCNN (super-resolution) method.
    This step now checks if the input data is a list (i.e. multiple slices) and
    processes each slice individually.
    """

    def __init__(self, scale: int, model: str) -> None:
        self.scale = scale
        self.model = self.__load_model(model)

    def apply(self, item: dict) -> dict:
        """
        Apply super-resolution to the image(s) stored in item["data"].
        
        :param item: Dictionary containing:
                     - "data": either a single numpy array or a list of 2D numpy arrays.
        :return: The updated item with the processed image(s).
        """
        data = item["data"]
        if isinstance(data, list):
            processed_slices = []
            for slice_img in data:
                processed_slice = self.__process_image(slice_img)
                processed_slices.append(processed_slice)
            item["data"] = processed_slices
        else:
            item["data"] = self.__process_image(data)
        return item

    def __process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image using FSRCNN.
        
        :param image: A 2D numpy array.
        :return: Processed image as a numpy array.
        """
        image_tensor = self.__image_to_tensor(image)
        with torch.no_grad():
            sr_image_tensor = self.model(image_tensor)
        sr_image = self.__tensor_to_image(sr_image_tensor)
        return sr_image

    def __image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        pil_image = Image.fromarray(image)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        image_tensor = transform(pil_image).unsqueeze(0)
        return image_tensor

    def __tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        tensor = tensor.squeeze(0)
        image = tensor.clamp(0, 1).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        # If the image is single-channel, remove the extra channel dimension.
        if image.shape[0] == 1:
            image = image[0]
        return image

    def __load_model(self, model_path: str) -> FSRCNN:
        """
        Load the FSRCNN model from a file.
        """
        model = FSRCNN(scale_factor=self.scale)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model


class PreprocessingPipeline:
    """
    A class representing a sequence of preprocessing steps.
    Processes all images in the dataset and saves the results in a unique output folder.
    """

    def __init__(self, dataset_path: str, steps: list, output_base_dir: str, config: dict) -> None:
        """
        :param dataset_path: Path to the dataset directory.
        :param steps: A list of PreprocessingStep instances.
        :param output_base_dir: Base directory for saving the processed dataset.
        :param config: Configuration dictionary with 'preprocessing' and 'model' keys.
        """
        self.dataset_path = dataset_path
        self.steps = steps
        self.output_base_dir = output_base_dir
        self.config = config

    @staticmethod
    def create_output_path(base_dir: str, config: dict) -> str:
        """
        Create a unique output path based on the configuration and a base directory.
        """
        preprocessing_parts = []
        for step in config.get("preprocessing", []):
            method = step.get("method", "none")
            if method.lower() == "none":
                continue
            if method.lower() == "fsrcnn":
                scale = step.get("params", {}).get("scale", "")
                preprocessing_parts.append(f"{method}_x{scale}")
            else:
                preprocessing_parts.append(method)
        model_type = config.get("model", {}).get("type", "model")
        folder_name = "_".join(preprocessing_parts) + "_" + model_type
        output_path = os.path.join(base_dir, folder_name)
        if os.path.exists(output_path):
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            new_folder_name = f"{folder_name}_{date_str}"
            output_path = os.path.join(base_dir, new_folder_name)
            counter = 1
            while os.path.exists(output_path):
                new_folder_name = f"{folder_name}_{date_str}_{counter}"
                output_path = os.path.join(base_dir, new_folder_name)
                counter += 1
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def process(self):
        """
        Process all images in the dataset by applying all preprocessing steps in sequence.
        For non-unet networks, processed images are saved (by stacking slices) as NIfTI files,
        while for unet the output structure is:
        
            dataset/
                images/
                    test/
                    train/
                    val/
                labels/
                    train/
                    val/
                    
        Training/validation items are randomly split (80% train, 20% val).
        """
        output_dataset_path = self.create_output_path(self.output_base_dir, self.config)
        print(f"Output will be saved to: {output_dataset_path}")

        is_unet = (self.config.get("model", {}).get("type", "").lower() == "unet")
        if is_unet:
            # Create unet-specific folder structure.
            images_dir = os.path.join(output_dataset_path, "images")
            labels_dir = os.path.join(output_dataset_path, "labels")
            for sub in ["train", "val", "test"]:
                os.makedirs(os.path.join(images_dir, sub), exist_ok=True)
            for sub in ["train", "val"]:
                os.makedirs(os.path.join(labels_dir, sub), exist_ok=True)
            random.seed(42)

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".nii.gz"):
                    # Determine if the file is a segmentation mask.
                    is_label = "MASK" in file.upper()
                    # For non-unet pipelines, skip mask files.
                    if not is_unet and is_label:
                        print(f"Skipping segmentation mask: {file}")
                        continue

                    input_file_path = os.path.join(root, file)
                    print(f"Processing file: {input_file_path}")
                    nii_img = nib.load(input_file_path)
                    item = {
                        "data": nii_img,  # Pass the nibabel image for flexibility.
                        "file_path": input_file_path,
                        "affine": nii_img.affine,
                        "header": nii_img.header,
                    }

                    # Process the item through all steps.
                    processed_item = self.__process_item(item)

                    if is_unet:
                        # Decide the split based on the input folder structure.
                        relative_path = os.path.relpath(root, self.dataset_path)
                        parts = relative_path.split(os.sep)
                        if len(parts) == 1:
                            # Test set: no labels.
                            split = "test"
                        else:
                            # Training/validation: random split.
                            split = "train" if random.random() < 0.8 else "val"

                        # Determine the base folder.
                        if is_label:
                            base_folder = os.path.join(output_dataset_path, "labels", split)
                        else:
                            base_folder = os.path.join(output_dataset_path, "images", split)

                        # Processed data is expected to be a list of slices.
                        slices = processed_item["data"]
                        for i, slice_img in enumerate(slices):
                            base_filename = os.path.basename(file).replace(".nii.gz", "")
                            out_filename = f"{base_filename}_slice_{i:03d}.png"
                            out_path = os.path.join(base_folder, out_filename)
                            cv2.imwrite(out_path, slice_img)
                            print(f"Saved slice: {out_path}")
                    else:
                        # For non-unet, stack slices and save as a NIfTI file.
                        data = processed_item["data"]
                        if isinstance(data, list):
                            data = np.stack(data, axis=-1)
                        relative_path = os.path.relpath(root, self.dataset_path)
                        output_dir = os.path.join(output_dataset_path, relative_path)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file_path = os.path.join(output_dir, file)
                        processed_nii = nib.Nifti1Image(data, processed_item["affine"], processed_item["header"])
                        nib.save(processed_nii, output_file_path)
                        print(f"Saved processed image: {output_file_path}")

    def __process_item(self, item: dict) -> dict:
        """
        Process the item by applying all preprocessing steps in sequence.
        
        :param item: Dictionary containing image data and metadata.
        :return: Processed item.
        """
        for step in self.steps:
            item = step.apply(item)
        return item


def create_pipeline_from_config(config: dict) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline from a configuration dictionary.
    
    :param config: Configuration dictionary defining the preprocessing steps.
    :return: An instance of PreprocessingPipeline.
    """
    steps = []
    dataset_path = config["dataset"]["input_path"]
    output_base_dir = config["dataset"]["output_path"]

    output_dataset_path = PreprocessingPipeline.create_output_path(output_base_dir, config)
    print(f"Output will be saved to: {output_dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

    for step_config in config.get("preprocessing", []):
        method = step_config.get("method")
        params = step_config.get("params", {})
        if method == "nifti_to_png":
            steps.append(NIfTItoPNG(output_dataset_path, **params))
        elif method == "fsrcnn":
            steps.append(FsrcnnStep(**params))
        else:
            print(f"Warning: Unknown method '{method}'.")
    return PreprocessingPipeline(dataset_path, steps, output_base_dir, config)
