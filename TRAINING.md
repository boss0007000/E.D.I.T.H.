# Model Training Guide

## Overview

This guide explains how to train custom models for the vehicle identification pipeline. The system currently runs with mock models for demonstration, but can be enhanced with real trained models.

## Models to Train

### 1. Vehicle Detection Model (YOLO)

**Purpose**: Detect vehicles in images

**Recommended Approach**: Fine-tune YOLOv8 on vehicle-specific datasets

**Dataset Requirements**:
- Images with vehicles in various poses, lighting conditions
- Bounding box annotations for cars, motorcycles, buses, trucks
- Recommended size: 10,000+ images

**Training Steps**:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='vehicle_detection.yaml',  # Dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU ID
)

# Save trained model
model.save('vehicle_detector.pt')
```

**Dataset Format** (vehicle_detection.yaml):
```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 4  # Number of classes
names: ['car', 'motorcycle', 'bus', 'truck']
```

### 2. Make/Model Classification Model

**Purpose**: Classify vehicle make and model

**Recommended Approach**: Fine-tune ResNet50/EfficientNet on vehicle make/model dataset

**Dataset Requirements**:
- Images of vehicles from consistent angles
- Balanced across different makes and models
- Recommended size: 100+ classes, 500+ images per class

**Training Steps**:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from torchvision import transforms

# Create model
model = timm.create_model('resnet50', pretrained=True, num_classes=200)
model = model.cuda()

# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = VehicleDataset(root='data/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), 'vehicle_classifier.pth')
```

### 3. Logo Detection Model

**Purpose**: Detect and recognize vehicle brand logos

**Recommended Approach**: Fine-tune YOLOv8 on logo dataset

**Dataset Requirements**:
- Images of vehicle logos from various angles
- Annotations for logo bounding boxes and brand labels
- Recommended size: 5,000+ images across 15-20 brands

**Training Steps**:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='logo_detection.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

model.save('logo_detector.pt')
```

### 4. OCR Model (Optional)

**Purpose**: Extract text from vehicle badges

**Approach**: Use pre-trained EasyOCR or Tesseract (no training needed)

If custom training is needed for better accuracy on vehicle-specific text:

```python
import easyocr

# Use pretrained models - works well for most cases
reader = easyocr.Reader(['en'])
```

### 5. Embedding Model (Optional)

**Purpose**: Create visual embeddings for similarity search

**Approach**: Use pre-trained CLIP model (works well without fine-tuning)

For custom fine-tuning on vehicle images:

```python
from transformers import CLIPModel, CLIPProcessor
import torch

# Load pretrained
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Fine-tune with contrastive learning on vehicle pairs
# (implementation depends on your dataset)
```

## Recommended Datasets

### Public Datasets

1. **Stanford Cars Dataset**
   - 16,185 images of 196 classes
   - Good for classification
   - http://ai.stanford.edu/~jkrause/cars/car_dataset.html

2. **CompCars Dataset**
   - 136,726 images covering 1,716 car models
   - Includes make, model, and year
   - http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

3. **VeRi-776 Dataset**
   - Vehicle re-identification dataset
   - Good for embedding learning
   - https://github.com/VehicleReId/VeRidataset

4. **COCO Dataset**
   - Contains vehicle detections
   - Good for object detection fine-tuning
   - https://cocodataset.org/

5. **Vehicle Make and Model Recognition Dataset (VMMR)**
   - 9,170 images of 291 classes
   - https://github.com/faezetta/VMMRdb

### Data Augmentation

Recommended augmentations for vehicle images:

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=1),
        A.MedianBlur(blur_limit=3, p=1),
        A.Blur(blur_limit=3, p=1),
    ], p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Integrating Trained Models

### 1. Update Configuration

Edit `vehicle_identification/configs/config.yaml`:

```yaml
detection:
  model: "path/to/vehicle_detector.pt"

classification:
  model_name: "resnet50"
  weights_path: "path/to/vehicle_classifier.pth"
  num_classes: 200

logo_detection:
  model: "path/to/logo_detector.pt"
```

### 2. Load Custom Models

```python
from vehicle_identification import VehicleIdentificationPipeline

pipeline = VehicleIdentificationPipeline(config_path='configs/custom_config.yaml')

# Load classifier with custom weights
pipeline.classifier.load_model(
    weights_path='models/vehicle_classifier.pth',
    class_names=my_class_names
)

# Load custom vehicle database
pipeline.load_database(
    database_path='data/vehicle_database.json',
    index_path='data/faiss_index.bin'
)

# Use pipeline
result = pipeline.identify('test_image.jpg')
```

## Building Vehicle Database

Create a comprehensive vehicle database for embedding retrieval:

```python
import json
import numpy as np
from vehicle_identification.modules.embedding import EmbeddingRetrieval

# Create database
database = []
embeddings = []

embedding_model = EmbeddingRetrieval()
embedding_model.load_model()

# Process each vehicle
for vehicle_info in vehicles:
    # Load vehicle image
    image = load_image(vehicle_info['image_path'])
    
    # Create embedding
    emb = embedding_model.create_embedding(image)
    
    database.append({
        'make': vehicle_info['make'],
        'model': vehicle_info['model'],
        'year_range': vehicle_info['year_range'],
        'image_path': vehicle_info['image_path']
    })
    embeddings.append(emb)

# Save database
with open('vehicle_database.json', 'w') as f:
    json.dump(database, f)

embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)

# Build index
embedding_model.build_index(database, embeddings)
embedding_model.save_index('faiss_index.bin', 'vehicle_database.json')
```

## Performance Optimization

### GPU Acceleration

```python
config = {
    'detection': {'device': 'cuda:0'},
    'classification': {'device': 'cuda:0'},
    'logo_detection': {'device': 'cuda:0'},
    'ocr': {'device': 'cuda:0'},
    'embedding': {'device': 'cuda:0'}
}

pipeline = VehicleIdentificationPipeline()
# Update devices after initialization
for module in [pipeline.detector, pipeline.classifier, pipeline.logo_detector, 
               pipeline.ocr, pipeline.embedding]:
    module.device = 'cuda:0'
```

### Batch Processing

```python
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = pipeline.batch_identify(images)
```

### Model Quantization

For faster inference on CPU:

```python
import torch

# Load model
model = torch.load('vehicle_classifier.pth')

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'vehicle_classifier_quantized.pth')
```

## Evaluation

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate on test set
predictions = []
ground_truth = []

for image, label in test_loader:
    result = pipeline.classifier.classify(image)
    predictions.append(result[0]['class_idx'])
    ground_truth.append(label)

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
report = classification_report(ground_truth, predictions)
cm = confusion_matrix(ground_truth, predictions)

print(f"Accuracy: {accuracy:.2%}")
print(report)
```

### End-to-End Metrics

```python
# Evaluate full pipeline
correct = 0
total = 0

for image, expected_make, expected_model in test_set:
    result = pipeline.identify(image)
    
    if result['best_match']['make'] == expected_make and \
       result['best_match']['model'] == expected_model:
        correct += 1
    total += 1

accuracy = correct / total
print(f"End-to-end accuracy: {accuracy:.2%}")
```

## Tips and Best Practices

1. **Start with pre-trained models** and fine-tune on your dataset
2. **Balance your dataset** across makes and models
3. **Use data augmentation** to increase dataset diversity
4. **Monitor training** with validation set
5. **Test on diverse conditions** (weather, lighting, angles)
6. **Iterate on fusion weights** to optimize for your use case
7. **Build a large vehicle database** for better retrieval results
8. **Use ensemble methods** for critical applications

## Support

For questions about model training, please open an issue on GitHub with the label `training`.
