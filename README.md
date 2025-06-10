# Automatic System for Semantic Segmentation of Multiple Sclerosis Lesions in MRI
This repository contains the code for the Master's Degree final project: "Semantic Segmentation of Multiple Sclerosis Lesions in Magnetic Resonance Images Enhanced by Super-Resolution" by **Samuel Sánchez Toca**.

# Table of Contents
- [Automatic System for Semantic Segmentation of Multiple Sclerosis Lesions in MRI](#automatic-system-for-semantic-segmentation-of-multiple-sclerosis-lesions-in-mri)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

# Description
This project implements a system for the semantic segmentation of multiple sclerosis lesions in MRI scans. It utilizes deep learning techniques, particularly convolutional neural networks (CNNs), to achieve high accuracy in lesion detection and segmentation.

It includes:
- FSRCNN: A super-resolution model to enhance MRI images.
- U-Net: A deep learning architecture for semantic segmentation.
- YOLOv11: A real-time object detection model adapted for lesion detection.

# Features
- End-to-end pipeline for training and evaluation
- Preprocessing modules with slice selection and super-resolution
- Support for U-Net and YOLOv11 architectures
- K-Fold cross-validation
- Comprehensive evaluation with Dice, IoU, Precision, and Recall

# Architecture
The system follows this pipeline:

MRI Volume → Slice Selection → Preprocessing → Super-Resolution (FSRCNN) → Model (U-Net / YOLOv11) → Postprocessing → Evaluation

![System Architecture](images/system_architecture.png)

# Installation
To set up the project, follow these steps:
1. Clone the repository:

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
# Usage
To use the system, follow these steps:
1. Use the dataset from the [ICPR 2024 Competition on Multiple Sclerosis Lesion Segmentation](https://iplab.dmi.unict.it/mfs/ms-les-seg/).

2. Run the script (self explanatory):
   ```bash
   python main.py
    ```

# Examples
The following images illustrate the results of the segmentation process:
![Example of Segmentation](examples/01_segmentation_example.png)

From left to right: 
- Iriginal MRI image
- Original lesion mask
- U-Net segmentation result
- YOLOv11 segmentation result

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.