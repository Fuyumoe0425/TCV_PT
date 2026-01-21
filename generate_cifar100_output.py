#!/usr/bin/env python3
"""
CIFAR-100 Dataset Sample Generator and Viewer
This script generates synthetic CIFAR-100 data to demonstrate output for sample 197.
"""

import numpy as np
import pickle
import os


def create_synthetic_cifar100():
    """Create synthetic CIFAR-100 dataset for demonstration."""
    data_dir = './data'
    cifar100_dir = os.path.join(data_dir, 'cifar-100-python')
    os.makedirs(cifar100_dir, exist_ok=True)
    
    # CIFAR-100 class names
    fine_label_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    coarse_label_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
        'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
        'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    # Create meta file
    meta = {
        b'fine_label_names': [name.encode('utf-8') for name in fine_label_names],
        b'coarse_label_names': [name.encode('utf-8') for name in coarse_label_names]
    }
    
    meta_file = os.path.join(cifar100_dir, 'meta')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    
    # Create training data
    # Generate 50000 samples (CIFAR-100 has 50000 training samples)
    np.random.seed(42)  # For reproducibility
    
    # Create synthetic image data
    data = np.random.randint(0, 256, size=(50000, 3072), dtype=np.uint8)
    
    # For sample 197, create a more interesting pattern
    # Let's make it look like it has structure
    sample_197_data = np.zeros(3072, dtype=np.uint8)
    
    # Create a gradient pattern for demonstration
    for i in range(3):  # RGB channels
        channel_start = i * 1024
        for y in range(32):
            for x in range(32):
                idx = channel_start + y * 32 + x
                # Create a diagonal gradient pattern
                value = int((x + y) * 255 / 63) if i == 0 else int((x * y) * 255 / 1024) if i == 1 else int(abs(x - y) * 255 / 31)
                sample_197_data[idx] = value
    
    data[197] = sample_197_data
    
    # Generate labels
    fine_labels = np.random.randint(0, 100, size=50000).tolist()
    coarse_labels = np.random.randint(0, 20, size=50000).tolist()
    
    # Set specific labels for sample 197
    fine_labels[197] = 43  # lion
    coarse_labels[197] = 8  # large_carnivores
    
    # Create training batch
    train_batch = {
        b'data': data,
        b'fine_labels': fine_labels,
        b'coarse_labels': coarse_labels,
        b'filenames': [f'train_{i}.png'.encode('utf-8') for i in range(50000)]
    }
    
    train_file = os.path.join(cifar100_dir, 'train')
    with open(train_file, 'wb') as f:
        pickle.dump(train_batch, f)
    
    print(f"Synthetic CIFAR-100 dataset created at {cifar100_dir}")
    return cifar100_dir


def load_cifar100_batch(file_path):
    """Load a CIFAR-100 batch file."""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


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
    print("CLASS INFORMATION:")
    print("-" * 80)
    print(f"Fine Label ID: {fine_label}")
    print(f"Fine Label Name: {fine_names[fine_label]}")
    print(f"Coarse Label ID: {coarse_label}")
    print(f"Coarse Label Name: {coarse_names[coarse_label]}")
    print()
    
    # Image data information
    print("IMAGE METADATA:")
    print("-" * 80)
    print(f"Image Shape: 32x32x3 (Height x Width x Channels)")
    print(f"Image Data Type: {image_data.dtype}")
    print(f"Image Data Range: [{image_data.min()}, {image_data.max()}]")
    print(f"Total Pixels: {len(image_data)} (32 x 32 x 3)")
    print()
    
    # Reshape image from flat array to 3D array
    image_3d = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
    
    # Channel statistics
    print("CHANNEL STATISTICS:")
    print("-" * 80)
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = image_3d[:, :, i]
        print(f"{channel} Channel:")
        print(f"  Mean: {channel_data.mean():.4f}")
        print(f"  Standard Deviation: {channel_data.std():.4f}")
        print(f"  Minimum: {channel_data.min()}")
        print(f"  Maximum: {channel_data.max()}")
        print(f"  Median: {np.median(channel_data):.4f}")
    print()
    
    # Raw pixel data (first few pixels)
    print("RAW IMAGE DATA (First 30 values, flattened):")
    print("-" * 80)
    print(f"{image_data[:30]}")
    print()
    
    # Image dimensions
    print("IMAGE DIMENSIONS:")
    print("-" * 80)
    print(f"Original flattened shape: {image_data.shape}")
    print(f"Reshaped 3D array: {image_3d.shape}")
    print(f"  - Height: {image_3d.shape[0]} pixels")
    print(f"  - Width: {image_3d.shape[1]} pixels")
    print(f"  - Channels: {image_3d.shape[2]} (RGB)")
    print()
    
    # Display sample of pixel values from different regions
    print("PIXEL VALUES FROM DIFFERENT REGIONS:")
    print("-" * 80)
    regions = [
        ("Top-left corner (2x2)", image_3d[0:2, 0:2, :]),
        ("Top-right corner (2x2)", image_3d[0:2, 30:32, :]),
        ("Bottom-left corner (2x2)", image_3d[30:32, 0:2, :]),
        ("Bottom-right corner (2x2)", image_3d[30:32, 30:32, :]),
        ("Center (2x2)", image_3d[15:17, 15:17, :])
    ]
    
    for region_name, region_data in regions:
        print(f"{region_name}:")
        print(f"  Shape: {region_data.shape}")
        print(f"  Values (RGB for each pixel):")
        for y in range(region_data.shape[0]):
            for x in range(region_data.shape[1]):
                rgb = region_data[y, x]
                print(f"    Position ({y},{x}): R={rgb[0]:3d}, G={rgb[1]:3d}, B={rgb[2]:3d}")
        print()
    
    # Complete data output - showing all channels separately
    print("COMPLETE IMAGE DATA BY CHANNEL:")
    print("-" * 80)
    print("(Showing first 5 rows of each channel for brevity)")
    print()
    
    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        print(f"{channel_name} Channel (32x32):")
        channel_data = image_3d[:, :, i]
        print(f"  First 5 rows:")
        for row_idx in range(min(5, 32)):
            print(f"    Row {row_idx:2d}: {channel_data[row_idx]}")
        print()
    
    # Histogram data
    print("HISTOGRAM DATA (Value distribution):")
    print("-" * 80)
    hist, bins = np.histogram(image_data, bins=16, range=(0, 256))
    print("Bins (16 equal ranges from 0-255):")
    for i in range(len(hist)):
        bin_start = int(bins[i])
        bin_end = int(bins[i+1])
        print(f"  [{bin_start:3d}-{bin_end:3d}): {hist[i]:5d} pixels ({hist[i]/len(image_data)*100:5.2f}%)")
    print()
    
    print("=" * 80)
    print("END OF ALL OUTPUT FOR SAMPLE #197")
    print("=" * 80)


def main():
    """Main function to create CIFAR-100 and display sample information."""
    sample_idx = 197
    
    # Create synthetic dataset
    cifar100_dir = create_synthetic_cifar100()
    
    # Load training data
    train_file = os.path.join(cifar100_dir, 'train')
    meta_file = os.path.join(cifar100_dir, 'meta')
    
    print("\nLoading CIFAR-100 training data...")
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
        print("\nImage visualization saved to: cifar100_sample_197.png")
    except ImportError:
        print("\nNote: matplotlib not installed. Image visualization skipped.")


if __name__ == '__main__':
    main()
