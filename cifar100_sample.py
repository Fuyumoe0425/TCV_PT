#!/usr/bin/env python3
"""
CIFAR-100 Dataset Sample Viewer
This script loads the CIFAR-100 dataset and displays all information for a specific sample.
"""

import numpy as np
import pickle
import os
import sys


def load_cifar100_batch(file_path):
    """Load a CIFAR-100 batch file."""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def download_cifar100_torchvision():
    """Download CIFAR-100 dataset using torchvision."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        
        print("Downloading CIFAR-100 dataset using torchvision...")
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True
        )
        
        cifar100_dir = os.path.join(data_dir, 'cifar-100-python')
        print(f"Dataset downloaded to {data_dir}")
        
        return cifar100_dir
    except ImportError:
        print("Error: torchvision not installed.")
        print("Please install with: pip install torchvision")
        return None


def download_cifar100():
    """Download CIFAR-100 dataset if not present."""
    data_dir = './data'
    cifar100_dir = os.path.join(data_dir, 'cifar-100-python')
    
    if os.path.exists(cifar100_dir):
        print(f"CIFAR-100 dataset already exists at {cifar100_dir}")
        return cifar100_dir
    
    # Try using torchvision first
    result = download_cifar100_torchvision()
    if result:
        return result
    
    # If torchvision doesn't work, try manual download
    try:
        import urllib.request
        import tarfile
        
        os.makedirs(data_dir, exist_ok=True)
        
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        tar_path = os.path.join(data_dir, 'cifar-100-python.tar.gz')
        
        print(f"Downloading CIFAR-100 dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        os.remove(tar_path)
        print(f"Dataset downloaded and extracted to {cifar100_dir}")
        
        return cifar100_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease manually download and extract the dataset:")
        print("1. Download: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
        print("2. Extract to: ./data/cifar-100-python/")
        return None


def get_class_names(meta_file):
    """Load class names from meta file."""
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    
    fine_label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
    coarse_label_names = [name.decode('utf-8') for name in meta[b'coarse_label_names']]
    
    return fine_label_names, coarse_label_names


def display_sample_info(sample_idx, data, fine_labels, coarse_labels, fine_names, coarse_names):
    """Display all information about a specific sample."""
    print("=" * 80)
    print(f"CIFAR-100 DATASET - SAMPLE #{sample_idx}")
    print("=" * 80)
    print()
    
    # Get sample data
    image_data = data[sample_idx]
    fine_label = fine_labels[sample_idx]
    coarse_label = coarse_labels[sample_idx]
    
    # Class information
    print(f"Fine Label ID: {fine_label}")
    print(f"Fine Label Name: {fine_names[fine_label]}")
    print(f"Coarse Label ID: {coarse_label}")
    print(f"Coarse Label Name: {coarse_names[coarse_label]}")
    print()
    
    # Image data information
    print(f"Image Shape: 32x32x3 (Height x Width x Channels)")
    print(f"Image Data Type: {image_data.dtype}")
    print(f"Image Data Range: [{image_data.min()}, {image_data.max()}]")
    print()
    
    # Reshape image from flat array to 3D array
    image_3d = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
    
    # Channel statistics
    print("Channel Statistics:")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = image_3d[:, :, i]
        print(f"  {channel} Channel:")
        print(f"    Mean: {channel_data.mean():.2f}")
        print(f"    Std: {channel_data.std():.2f}")
        print(f"    Min: {channel_data.min()}")
        print(f"    Max: {channel_data.max()}")
    print()
    
    # Raw pixel data (first few pixels)
    print("Raw Image Data (first 10 pixels, flattened):")
    print(f"  {image_data[:10]}")
    print()
    
    # Image dimensions
    print("Image Data Dimensions:")
    print(f"  Original shape: {image_data.shape}")
    print(f"  Reshaped 3D: {image_3d.shape}")
    print()
    
    # All pixel values (optional - can be very long)
    print("Complete Image Data Array:")
    print(f"  Total pixels: {len(image_data)}")
    print(f"  Complete array shape: {image_data.shape}")
    print()
    
    # Display sample of pixel values from different regions
    print("Sample Pixel Values from Different Regions:")
    regions = [
        ("Top-left corner", image_3d[0:2, 0:2, :]),
        ("Top-right corner", image_3d[0:2, 30:32, :]),
        ("Bottom-left corner", image_3d[30:32, 0:2, :]),
        ("Bottom-right corner", image_3d[30:32, 30:32, :]),
        ("Center", image_3d[15:17, 15:17, :])
    ]
    
    for region_name, region_data in regions:
        print(f"  {region_name}:")
        print(f"    Shape: {region_data.shape}")
        print(f"    Values:\n{region_data}")
    print()
    
    print("=" * 80)
    print("END OF OUTPUT")
    print("=" * 80)


def main():
    """Main function to load CIFAR-100 and display sample information."""
    sample_idx = 197
    
    # Download dataset if needed
    cifar100_dir = download_cifar100()
    
    if not cifar100_dir or not os.path.exists(cifar100_dir):
        print("\nDataset not available. Exiting.")
        return
    
    # Load training data
    train_file = os.path.join(cifar100_dir, 'train')
    meta_file = os.path.join(cifar100_dir, 'meta')
    
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        print("Please ensure the CIFAR-100 dataset is properly extracted.")
        return
    
    print("Loading CIFAR-100 training data...")
    train_batch = load_cifar100_batch(train_file)
    
    # Extract data
    data = train_batch[b'data']
    fine_labels = train_batch[b'fine_labels']
    coarse_labels = train_batch[b'coarse_labels']
    
    # Load class names
    fine_names, coarse_names = get_class_names(meta_file)
    
    print(f"Total training samples: {len(data)}")
    print(f"Number of fine classes: {len(fine_names)}")
    print(f"Number of coarse classes: {len(coarse_names)}")
    print()
    
    # Display information for sample 197
    display_sample_info(sample_idx, data, fine_labels, coarse_labels, fine_names, coarse_names)
    
    # Save the image to a file if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        image_data = data[sample_idx]
        image_3d = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image_3d)
        plt.title(f"CIFAR-100 Sample #{sample_idx}: {fine_names[fine_labels[sample_idx]]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('cifar100_sample_197.png', dpi=150, bbox_inches='tight')
        print("\nImage saved to: cifar100_sample_197.png")
    except ImportError:
        print("\nNote: matplotlib not installed. Image visualization skipped.")
        print("Install with: pip install matplotlib")


if __name__ == '__main__':
    main()
