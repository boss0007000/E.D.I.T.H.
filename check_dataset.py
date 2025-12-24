#!/usr/bin/env python3
"""
Utility to check dataset structure and provide statistics

Usage:
    python check_dataset.py --dataset dataset
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


def check_dataset(dataset_path: str):
    """Check dataset structure and provide statistics"""
    
    print("="*70)
    print("E.D.I.T.H. Dataset Structure Checker")
    print("="*70)
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset folder not found: {dataset_path}")
        print("\nExpected structure:")
        print("dataset/")
        print("├── acura_cl_1997/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("├── honda_civic_2005/")
        print("│   └── ...")
        print("└── ...")
        return False
    
    print(f"\n📁 Dataset path: {dataset_path.absolute()}")
    
    # Get all class folders
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if not class_folders:
        print(f"\n❌ Error: No class folders found in {dataset_path}")
        return False
    
    print(f"\n✅ Found {len(class_folders)} vehicle classes\n")
    
    # Analyze each class
    class_stats = []
    total_images = 0
    makes = defaultdict(int)
    years = defaultdict(int)
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Count images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_folder.glob(ext)))
        
        num_images = len(image_files)
        total_images += num_images
        
        # Parse folder name
        parts = class_name.split('_')
        if len(parts) >= 3:
            make = parts[0].capitalize()
            model = '_'.join(parts[1:-1]).replace('_', ' ').title()
            year = parts[-1]
        elif len(parts) == 2:
            make = parts[0].capitalize()
            model = parts[1].replace('_', ' ').title()
            year = 'Unknown'
        else:
            make = class_name.capitalize()
            model = 'Unknown'
            year = 'Unknown'
        
        makes[make] += 1
        if year != 'Unknown':
            years[year] += 1
        
        class_stats.append({
            'folder': class_name,
            'make': make,
            'model': model,
            'year': year,
            'images': num_images
        })
    
    # Print summary
    print("-"*70)
    print(f"{'Folder Name':<30} {'Make':<15} {'Model':<20} {'Images':>5}")
    print("-"*70)
    
    for stats in class_stats[:20]:  # Show first 20
        print(f"{stats['folder']:<30} {stats['make']:<15} {stats['model']:<20} {stats['images']:>5}")
    
    if len(class_stats) > 20:
        print(f"... and {len(class_stats) - 20} more classes")
    
    print("-"*70)
    
    # Statistics
    print("\n📊 Dataset Statistics:")
    print(f"  Total classes: {len(class_stats)}")
    print(f"  Total images: {total_images}")
    print(f"  Average images per class: {total_images / len(class_stats):.1f}")
    print(f"  Unique makes: {len(makes)}")
    
    # Check for issues
    print("\n🔍 Data Quality Check:")
    
    # Check for classes with too few images
    min_images = 5
    low_image_classes = [s for s in class_stats if s['images'] < min_images]
    if low_image_classes:
        print(f"  ⚠️  Warning: {len(low_image_classes)} classes have fewer than {min_images} images")
        print(f"      This may affect training quality")
    else:
        print(f"  ✅ All classes have at least {min_images} images")
    
    # Check for balanced dataset
    max_images = max(s['images'] for s in class_stats)
    min_images_found = min(s['images'] for s in class_stats)
    imbalance_ratio = max_images / max(min_images_found, 1)
    
    if imbalance_ratio > 10:
        print(f"  ⚠️  Warning: Dataset is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        print(f"      Consider data augmentation or rebalancing")
    else:
        print(f"  ✅ Dataset is reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
    
    # Most common makes
    print(f"\n🏭 Top Makes:")
    top_makes = sorted(makes.items(), key=lambda x: x[1], reverse=True)[:5]
    for make, count in top_makes:
        print(f"  {make}: {count} models")
    
    print("\n" + "="*70)
    print("✅ Dataset structure is valid!")
    print("="*70)
    
    print("\nYou can now train the model with:")
    print(f"  python train.py --dataset {dataset_path} --epochs 50")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Check E.D.I.T.H. dataset structure')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to dataset folder (default: dataset)')
    
    args = parser.parse_args()
    
    check_dataset(args.dataset)


if __name__ == '__main__':
    main()
