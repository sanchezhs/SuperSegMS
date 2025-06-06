# Multiple Sclerosis Lesion Segmentation in MRI using Deep Learning

Master's Thesis focused on developing an automatic system for semantic segmentation of multiple sclerosis (MS) lesions from magnetic resonance imaging (MRI). Computer vision techniques and deep neural networks are used, with emphasis on architectures such as U-Net and YOLOv8/YOLOv11, as well as resolution enhancement methods like FSRCNN.

# Implemented Models

- **U-Net** with ResNet34 encoder (`segmentation_models_pytorch`)
- **YOLOv8 / YOLOv11** for fast and accurate 2D segmentation
- **FSRCNN** to enhance the input resolution of MRI images

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/sanchezhs/med-seg-tfm.git
```

2. Navigate to the project directory:
```bash
cd med-seg-tfm
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pretrained model weights:
    - YOLO: [Download](https://docs.ultralytics.com/models/)
    - FSRCNN: [Download](https://github.com/yjn870/FSRCNN-pytorch)

5. Run the script (self-explanatory):
```bash
python main.py --help
```