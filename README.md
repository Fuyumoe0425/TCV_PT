# TCV_PT

## CIFAR-100 Dataset Sample Viewer

This project provides a script to load and display all information about a specific sample from the CIFAR-100 dataset.

### Requirements

- Python 3.6+
- numpy
- matplotlib (optional, for image visualization)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the script to display all information about sample #197 from the CIFAR-100 dataset:

```bash
python cifar100_sample.py
```

The script will:
1. Automatically download the CIFAR-100 dataset if not present
2. Load the training data
3. Display comprehensive information about sample #197, including:
   - Fine and coarse labels (ID and name)
   - Image dimensions and data type
   - Pixel value statistics (mean, std, min, max) for each color channel
   - Sample pixel values from different regions of the image
   - Complete data array information
4. Save a visualization of the image to `cifar100_sample_197.png`

### Output

The script provides detailed output including:
- Label information (fine-grained and coarse-grained classification)
- Image metadata (shape, data type, value ranges)
- Channel-wise statistics (RGB)
- Sample pixel values from different regions
- Complete image data information

### CIFAR-100 Dataset

CIFAR-100 is a dataset of 60,000 32x32 color images in 100 classes, with 600 images per class. The 100 classes are grouped into 20 superclasses. Each image comes with a "fine" label (the class) and a "coarse" label (the superclass).

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html