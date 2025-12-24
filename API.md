# API Reference

Complete API documentation for the Vehicle Identification System.

## Main Pipeline

### VehicleIdentificationPipeline

Main class for orchestrating the complete identification pipeline.

```python
from vehicle_identification import VehicleIdentificationPipeline

pipeline = VehicleIdentificationPipeline(config_path='path/to/config.yaml')
```

#### Constructor

```python
VehicleIdentificationPipeline(config_path: str = None)
```

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file. If None, uses default config.

#### Methods

##### identify()

Identify vehicle in a single image.

```python
result = pipeline.identify(image)
```

**Parameters:**
- `image` (str | np.ndarray | PIL.Image): Input image (file path, numpy array, or PIL Image)

**Returns:**
- `dict`: Identification results with structure:
  ```python
  {
      'status': 'success',  # or 'no_vehicle_detected'
      'best_match': {
          'make': str,
          'model': str,
          'year_range': str,
          'confidence': float
      },
      'alternatives': [
          {
              'make': str,
              'model': str,
              'year_range': str,
              'confidence': float
          }
      ],
      'evidence': {
          'classification': list,
          'logo': list,
          'ocr': list,
          'embedding': list
      },
      'component_scores': {
          'classification': float,
          'logo': float,
          'ocr': float,
          'embedding': float
      },
      'detection_info': dict
  }
  ```

##### batch_identify()

Identify vehicles in multiple images.

```python
results = pipeline.batch_identify(images)
```

**Parameters:**
- `images` (list): List of images (same format as `identify()`)

**Returns:**
- `list[dict]`: List of identification results

##### load_database()

Load vehicle database for embedding retrieval.

```python
pipeline.load_database(
    database_path='data/vehicles.json',
    index_path='data/index.bin',
    embeddings=None
)
```

**Parameters:**
- `database_path` (str, optional): Path to vehicle database JSON
- `index_path` (str, optional): Path to pre-built search index
- `embeddings` (np.ndarray, optional): Pre-computed embeddings

---

## Individual Modules

### VehicleDetector

Detects vehicles in images using YOLO.

```python
from vehicle_identification.modules import VehicleDetector

detector = VehicleDetector(
    model_name='yolov8n',
    confidence_threshold=0.3,
    iou_threshold=0.5,
    device='cpu',
    target_classes=[2, 3, 5, 7]
)
detector.load_model()
```

#### Methods

##### detect()

```python
detections = detector.detect(image)
```

**Returns:**
```python
[
    {
        'bbox': [x1, y1, x2, y2],
        'class': int,
        'confidence': float,
        'class_name': str
    }
]
```

##### crop_vehicle()

```python
cropped = detector.crop_vehicle(image, bbox, padding=0.1)
```

---

### VehicleClassifier

Classifies vehicle make and model.

```python
from vehicle_identification.modules import VehicleClassifier

classifier = VehicleClassifier(
    model_name='resnet50',
    num_classes=100,
    confidence_threshold=0.1,
    top_k=5,
    device='cpu'
)
classifier.load_model(weights_path='path/to/weights.pth', class_names=classes)
```

#### Methods

##### classify()

```python
predictions = classifier.classify(image)
```

**Returns:**
```python
[
    {
        'class_idx': int,
        'class_name': str,
        'make': str,
        'model': str,
        'confidence': float,
        'year_range': str
    }
]
```

---

### LogoDetector

Detects and recognizes vehicle brand logos.

```python
from vehicle_identification.modules import LogoDetector

logo_detector = LogoDetector(
    model_name='yolov8n',
    confidence_threshold=0.25,
    device='cpu'
)
logo_detector.load_model(weights_path='path/to/logo_model.pt')
```

#### Methods

##### detect_logos()

```python
logos = logo_detector.detect_logos(image)
```

**Returns:**
```python
[
    {
        'bbox': [x1, y1, x2, y2],
        'brand': str,
        'confidence': float
    }
]
```

---

### VehicleOCR

Extracts text from vehicle images.

```python
from vehicle_identification.modules import VehicleOCR

ocr = VehicleOCR(
    reader='easyocr',  # or 'tesseract'
    languages=['en'],
    confidence_threshold=0.5,
    device='cpu'
)
ocr.load_reader()
```

#### Methods

##### extract_text()

```python
text_results = ocr.extract_text(image, preprocess=True)
```

**Returns:**
```python
[
    {
        'text': str,
        'confidence': float,
        'bbox': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
]
```

##### extract_model_info()

```python
model_info = ocr.extract_model_info(text_results)
```

**Returns:** `list[str]` - Potential model badges/names

##### extract_brand_info()

```python
brands = ocr.extract_brand_info(text_results)
```

**Returns:** `list[str]` - Potential brand names

---

### EmbeddingRetrieval

Creates embeddings and performs similarity search.

```python
from vehicle_identification.modules import EmbeddingRetrieval

embedding = EmbeddingRetrieval(
    model_name='sentence-transformers/clip-ViT-B-32',
    dimension=512,
    device='cpu',
    index_type='faiss'  # or 'hnswlib'
)
embedding.load_model()
```

#### Methods

##### create_embedding()

```python
emb = embedding.create_embedding(image)
```

**Returns:** `np.ndarray` - Normalized embedding vector

##### build_index()

```python
embedding.build_index(database, embeddings=None)
```

##### search()

```python
results = embedding.search(query_embedding, top_k=10)
```

**Returns:**
```python
[
    (index, similarity_score),
    ...
]
```

##### retrieve()

```python
vehicles = embedding.retrieve(query_embedding, top_k=10)
```

**Returns:**
```python
[
    {
        'make': str,
        'model': str,
        'year_range': str,
        'similarity_score': float
    }
]
```

---

### ResultFusion

Combines results from multiple sources.

```python
from vehicle_identification.modules import ResultFusion

fusion = ResultFusion(
    weights={
        'classification': 0.4,
        'logo': 0.2,
        'ocr': 0.1,
        'embedding': 0.3
    },
    final_top_k=5
)
```

#### Methods

##### fuse_results()

```python
fused = fusion.fuse_results(
    classification_results=cls_results,
    logo_results=logo_results,
    ocr_results=ocr_results,
    embedding_results=emb_results
)
```

**Returns:** `list[dict]` - Ranked vehicle matches

##### format_final_results()

```python
final = fusion.format_final_results(fused_results)
```

**Returns:** Formatted result dictionary (same as `VehicleIdentificationPipeline.identify()`)

---

## Utility Functions

### Image Utilities

```python
from vehicle_identification.utils import (
    load_image,
    save_image,
    resize_image,
    visualize_detection,
    create_comparison_image,
    format_confidence
)
```

#### load_image()

```python
image = load_image('path/to/image.jpg')
```

**Returns:** `np.ndarray` - RGB image

#### save_image()

```python
save_image(image, 'output.jpg')
```

#### resize_image()

```python
resized = resize_image(image, size=(800, 600))
```

#### visualize_detection()

```python
vis_image = visualize_detection(
    image,
    bbox=[x1, y1, x2, y2],
    label='Toyota Camry',
    confidence=0.85,
    color=(0, 255, 0)
)
```

#### format_confidence()

```python
conf_str = format_confidence(0.856)  # Returns "85.6%"
```

---

## Configuration

Configuration file format (YAML):

```yaml
detection:
  model: "yolov8n"
  confidence_threshold: 0.3
  iou_threshold: 0.5
  device: "cpu"
  target_classes: [2, 3, 5, 7]

classification:
  model_name: "resnet50"
  num_classes: 100
  confidence_threshold: 0.1
  top_k: 5
  device: "cpu"

logo_detection:
  model: "yolov8n"
  confidence_threshold: 0.25
  device: "cpu"

ocr:
  reader: "easyocr"
  languages: ["en"]
  confidence_threshold: 0.5
  device: "cpu"

embedding:
  model_name: "sentence-transformers/clip-ViT-B-32"
  dimension: 512
  device: "cpu"
  index_type: "faiss"

retrieval:
  top_k: 10
  similarity_metric: "cosine"

fusion:
  weights:
    classification: 0.4
    logo: 0.2
    ocr: 0.1
    embedding: 0.3
  final_top_k: 5
```

---

## Examples

### Basic Usage

```python
from vehicle_identification import VehicleIdentificationPipeline

# Initialize
pipeline = VehicleIdentificationPipeline()
pipeline.load_database()

# Identify vehicle
result = pipeline.identify('car.jpg')

# Print results
if result['status'] == 'success':
    best = result['best_match']
    print(f"{best['make']} {best['model']} ({best['year_range']})")
    print(f"Confidence: {best['confidence']:.1%}")
```

### Custom Configuration

```python
pipeline = VehicleIdentificationPipeline(config_path='custom_config.yaml')
pipeline.load_database(database_path='custom_db.json')
result = pipeline.identify('car.jpg')
```

### Batch Processing

```python
images = ['car1.jpg', 'car2.jpg', 'car3.jpg']
results = pipeline.batch_identify(images)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['best_match']['make']} {result['best_match']['model']}")
```

### Using Individual Modules

```python
from vehicle_identification.modules import VehicleDetector, VehicleClassifier

# Detection only
detector = VehicleDetector()
detector.load_model()
detections = detector.detect(image)

# Classification only
classifier = VehicleClassifier()
classifier.load_model()
predictions = classifier.classify(cropped_vehicle)
```

---

## Error Handling

The pipeline handles errors gracefully and uses mock models when dependencies are missing:

```python
result = pipeline.identify('image.jpg')

if result['status'] == 'success':
    # Process successful identification
    pass
elif result['status'] == 'no_vehicle_detected':
    print("No vehicle found in image")
else:
    print(f"Error: {result.get('message', 'Unknown error')}")
```

---

## Performance Considerations

### GPU Acceleration

Set `device: "cuda"` in config for GPU acceleration:

```yaml
detection:
  device: "cuda:0"
classification:
  device: "cuda:0"
# ... etc
```

### Batch Processing

Process multiple images together for better throughput:

```python
results = pipeline.batch_identify(images)
```

### Model Selection

- Use `yolov8n` for speed, `yolov8x` for accuracy
- Use `resnet50` for balance, `efficientnet_b7` for accuracy
- Adjust `top_k` and confidence thresholds based on needs

---

For more examples, see the `example.py` and `test_pipeline.py` scripts.
