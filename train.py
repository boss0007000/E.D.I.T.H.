#!/usr/bin/env python3
"""
Training script for E.D.I.T.H. vehicle classification model

Supports dataset structure:
dataset/
├── acura_cl_1997/
│   ├── image1.jpg
│   └── ...
├── honda_civic_2005/
│   ├── image1.jpg
│   └── ...
└── ...

Each folder name should be: <make>_<model>_<year>
"""

import os
import argparse
from pathlib import Path
from typing import List
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np


class VehicleDataset(Dataset):
    """Dataset loader for vehicle images organized by folders"""
    
    def __init__(self, root_dir: str, transform=None, split: str = 'train', train_ratio: float = 0.8):
        """
        Args:
            root_dir: Path to dataset folder
            transform: Image transformations
            split: 'train' or 'val'
            train_ratio: Ratio of images to use for training
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        # Load all images and labels
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from folder structure"""
        print(f"Loading dataset from {self.root_dir}...")
        
        # Get all class folders
        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {self.root_dir}")
        
        self.classes = [folder.name for folder in class_folders]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} vehicle classes")
        
        # Load all images
        all_samples = []
        for class_folder in class_folders:
            class_name = class_folder.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_folder.glob(ext)))
            
            # Add samples
            for img_path in image_files:
                all_samples.append((str(img_path), class_idx))
        
        if not all_samples:
            raise ValueError(f"No images found in {self.root_dir}")
        
        # Shuffle samples with fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(all_samples)
        
        # Split into train/val
        split_idx = int(len(all_samples) * self.train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
            print(f"Training samples: {len(self.samples)}")
        else:
            self.samples = all_samples[split_idx:]
            print(f"Validation samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_info(self) -> List[dict]:
        """Get information about classes parsed from folder names"""
        class_info = []
        
        for class_name in self.classes:
            # Try to parse: make_model_year
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
            
            class_info.append({
                'folder_name': class_name,
                'make': make,
                'model': model,
                'year': year
            })
        
        return class_info


def train_model(args):
    """Train the vehicle classification model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    train_dataset = VehicleDataset(
        root_dir=args.dataset,
        transform=train_transform,
        split='train',
        train_ratio=args.train_ratio
    )
    
    val_dataset = VehicleDataset(
        root_dir=args.dataset,
        transform=val_transform,
        split='val',
        train_ratio=args.train_ratio
    )
    
    # Save class information
    class_info = train_dataset.get_class_info()
    class_names = [info['folder_name'] for info in class_info]
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    with open('models/class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nSaved {len(class_names)} class names to models/class_names.json")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Number of classes: {len(class_names)}")
    
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(class_names)
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/vehicle_classifier.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'models/checkpoint_epoch_{epoch + 1}.pth')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: models/vehicle_classifier.pth")
    print(f"Class names saved to: models/class_names.json")
    print(f"Class info saved to: models/class_info.json")


def main():
    parser = argparse.ArgumentParser(description='Train E.D.I.T.H. vehicle classifier')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to dataset folder (default: dataset)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of images for training (default: 0.8)')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model architecture (default: resnet50)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU if available (default: True)')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset folder not found: {args.dataset}")
        print("\nExpected structure:")
        print("dataset/")
        print("├── acura_cl_1997/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("├── honda_civic_2005/")
        print("│   └── ...")
        print("└── ...")
        return
    
    print("="*70)
    print("E.D.I.T.H. Vehicle Classifier Training")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"GPU: {'Yes' if torch.cuda.is_available() and args.gpu else 'No'}")
    
    train_model(args)


if __name__ == '__main__':
    main()
