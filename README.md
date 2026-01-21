# TCV_PT

## CIFAR-100 Dataset Sample Viewer

This project provides a script to generate synthetic CIFAR-100 dataset and display all information about a specific sample (sample #197).

### Requirements

- Python 3.6+
- numpy
- matplotlib (optional, for image visualization)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the script to generate synthetic CIFAR-100 data and display all information about sample #197:

```bash
python generate_cifar100_output.py
```

The script will:
1. Generate a synthetic CIFAR-100 dataset (mimicking the real dataset structure)
2. Load the training data
3. Display comprehensive information about sample #197, including:
   - Fine and coarse labels (ID and name)
   - Image dimensions and data type
   - Pixel value statistics (mean, std, min, max, median) for each color channel
   - Raw image data (first 30 values)
   - Sample pixel values from different regions of the image (corners and center)
   - Complete data array information for all channels
   - Histogram showing pixel value distribution
4. Save a visualization of the image to `cifar100_sample_197.png`

### Output

The script provides comprehensive output including:
- **Class Information**: Fine-grained label (lion) and coarse-grained label (large_carnivores)
- **Image Metadata**: Shape, data type, value ranges
- **Channel Statistics**: Mean, std, min, max, median for RGB channels
- **Raw Pixel Data**: First 30 values from the flattened array
- **Regional Pixel Samples**: 2x2 pixel blocks from 5 different regions (corners and center)
- **Complete Channel Data**: First 5 rows of each color channel (32x32 pixels)
- **Histogram**: Distribution of pixel values across 16 bins

### Sample #197 Details

- **Fine Label**: lion (ID: 43)
- **Coarse Label**: large_carnivores (ID: 8)
- **Image Size**: 32×32 pixels, 3 channels (RGB)
- **Total Values**: 3,072 values (32 × 32 × 3)

### CIFAR-100 Dataset

CIFAR-100 is a dataset of 60,000 32x32 color images in 100 classes, with 600 images per class. The 100 classes are grouped into 20 superclasses. Each image comes with a "fine" label (the class) and a "coarse" label (the superclass).

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

### Alternative Script

An alternative script `cifar100_sample.py` is also available that attempts to download the real CIFAR-100 dataset from the internet if you have connectivity.