"""
Main Pipeline
Orchestrates the complete vehicle identification pipeline
"""
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image
import yaml
import os

from .modules.detector import VehicleDetector
from .modules.classifier import VehicleClassifier
from .modules.logo_detector import LogoDetector
from .modules.ocr import VehicleOCR
from .modules.embedding import EmbeddingRetrieval
from .modules.fusion import ResultFusion


class VehicleIdentificationPipeline:
    """Complete vehicle identification pipeline"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize all modules
        self.detector = None
        self.classifier = None
        self.logo_detector = None
        self.ocr = None
        self.embedding = None
        self.fusion = None
        
        self._initialize_modules()
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'detection': {
                    'model': 'yolov8n',
                    'confidence_threshold': 0.3,
                    'iou_threshold': 0.5,
                    'device': 'cpu',
                    'target_classes': [2, 3, 5, 7]
                },
                'classification': {
                    'model_name': 'resnet50',
                    'num_classes': 100,
                    'confidence_threshold': 0.1,
                    'top_k': 5,
                    'device': 'cpu'
                },
                'logo_detection': {
                    'model': 'yolov8n',
                    'confidence_threshold': 0.25,
                    'device': 'cpu'
                },
                'ocr': {
                    'reader': 'easyocr',
                    'languages': ['en'],
                    'confidence_threshold': 0.5,
                    'device': 'cpu'
                },
                'embedding': {
                    'model_name': 'sentence-transformers/clip-ViT-B-32',
                    'dimension': 512,
                    'device': 'cpu',
                    'index_type': 'faiss'
                },
                'fusion': {
                    'weights': {
                        'classification': 0.4,
                        'logo': 0.2,
                        'ocr': 0.1,
                        'embedding': 0.3
                    },
                    'final_top_k': 5
                }
            }
        
        return config
    
    def _initialize_modules(self):
        """Initialize all pipeline modules"""
        print("Initializing vehicle identification pipeline...")
        
        # Detector
        det_cfg = self.config['detection']
        self.detector = VehicleDetector(
            model_name=det_cfg['model'],
            confidence_threshold=det_cfg['confidence_threshold'],
            iou_threshold=det_cfg['iou_threshold'],
            device=det_cfg['device'],
            target_classes=det_cfg['target_classes']
        )
        self.detector.load_model()
        
        # Classifier
        cls_cfg = self.config['classification']
        self.classifier = VehicleClassifier(
            model_name=cls_cfg['model_name'],
            num_classes=cls_cfg['num_classes'],
            confidence_threshold=cls_cfg['confidence_threshold'],
            top_k=cls_cfg['top_k'],
            device=cls_cfg['device']
        )
        self.classifier.load_model()
        
        # Logo detector
        logo_cfg = self.config['logo_detection']
        self.logo_detector = LogoDetector(
            model_name=logo_cfg['model'],
            confidence_threshold=logo_cfg['confidence_threshold'],
            device=logo_cfg['device']
        )
        self.logo_detector.load_model()
        
        # OCR
        ocr_cfg = self.config['ocr']
        self.ocr = VehicleOCR(
            reader=ocr_cfg['reader'],
            languages=ocr_cfg['languages'],
            confidence_threshold=ocr_cfg['confidence_threshold'],
            device=ocr_cfg['device']
        )
        self.ocr.load_reader()
        
        # Embedding
        emb_cfg = self.config['embedding']
        self.embedding = EmbeddingRetrieval(
            model_name=emb_cfg['model_name'],
            dimension=emb_cfg['dimension'],
            device=emb_cfg['device'],
            index_type=emb_cfg['index_type']
        )
        self.embedding.load_model()
        
        # Fusion
        fusion_cfg = self.config['fusion']
        self.fusion = ResultFusion(
            weights=fusion_cfg['weights'],
            final_top_k=fusion_cfg['final_top_k']
        )
        
        print("Pipeline initialized successfully!")
    
    def load_database(self, database_path: str = None, 
                     index_path: str = None,
                     embeddings: np.ndarray = None):
        """
        Load vehicle database and build search index
        
        Args:
            database_path: Path to vehicle database
            index_path: Path to pre-built index (optional)
            embeddings: Pre-computed embeddings (optional)
        """
        if database_path and os.path.exists(database_path):
            import json
            with open(database_path, 'r') as f:
                database = json.load(f)
        else:
            # Create mock database
            database = self._create_mock_database()
        
        if index_path and os.path.exists(index_path):
            # Load existing index
            self.embedding.load_index(index_path, database_path)
        else:
            # Build new index
            self.embedding.build_index(database, embeddings)
    
    def _create_mock_database(self) -> List[Dict]:
        """Create a mock vehicle database for demonstration"""
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW']
        models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander'],
            'Honda': ['Accord', 'Civic', 'CR-V', 'Pilot'],
            'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape'],
            'Chevrolet': ['Silverado', 'Malibu', 'Equinox', 'Tahoe'],
            'BMW': ['3 Series', '5 Series', 'X5', 'X3']
        }
        
        database = []
        for make in makes:
            for model in models[make]:
                for year_start in [2018, 2020, 2022]:
                    database.append({
                        'make': make,
                        'model': model,
                        'year_range': f'{year_start}-{year_start + 2}'
                    })
        
        return database
    
    def identify(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Identify vehicle in image
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Identification results
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            img_array = np.array(img)
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image
        
        print("\n" + "="*60)
        print("VEHICLE IDENTIFICATION PIPELINE")
        print("="*60)
        
        # Stage 1: Vehicle Detection
        print("\n[Stage 1/6] Detecting vehicles...")
        detections = self.detector.detect(img_array)
        print(f"Found {len(detections)} vehicle(s)")
        
        if not detections:
            return {
                'status': 'no_vehicle_detected',
                'message': 'No vehicle detected in the image',
                'best_match': None,
                'alternatives': [],
                'evidence': {}
            }
        
        # Use the first (most confident) detection
        detection = detections[0]
        
        # Stage 2: Crop Vehicle
        print("\n[Stage 2/6] Cropping vehicle region...")
        cropped = self.detector.crop_vehicle(img_array, detection['bbox'])
        print(f"Cropped vehicle: {cropped.shape}")
        
        # Stage 3: Classification
        print("\n[Stage 3/6] Classifying make/model...")
        classification_results = self.classifier.classify(cropped)
        print(f"Top prediction: {classification_results[0]['class_name']} "
              f"(confidence: {classification_results[0]['confidence']:.3f})")
        
        # Stage 4: Logo Detection
        print("\n[Stage 4/6] Detecting logos...")
        logo_results = self.logo_detector.detect_logos(cropped)
        if logo_results:
            print(f"Detected logo: {logo_results[0]['brand']} "
                  f"(confidence: {logo_results[0]['confidence']:.3f})")
        else:
            print("No logos detected")
        
        # Stage 5: OCR
        print("\n[Stage 5/6] Extracting text (OCR)...")
        ocr_results = self.ocr.extract_text(cropped)
        if ocr_results:
            print(f"Extracted {len(ocr_results)} text region(s)")
            for result in ocr_results[:3]:  # Show top 3
                print(f"  - '{result['text']}' (confidence: {result['confidence']:.3f})")
        else:
            print("No text extracted")
        
        # Stage 6: Embedding & Retrieval
        print("\n[Stage 6/6] Searching similar vehicles...")
        query_embedding = self.embedding.create_embedding(cropped)
        embedding_results = self.embedding.retrieve(query_embedding, top_k=10)
        print(f"Retrieved {len(embedding_results)} similar vehicles")
        
        # Fusion
        print("\n[Fusion] Combining all results...")
        fused_results = self.fusion.fuse_results(
            classification_results=classification_results,
            logo_results=logo_results,
            ocr_results=ocr_results,
            embedding_results=embedding_results
        )
        
        final_results = self.fusion.format_final_results(fused_results)
        final_results['status'] = 'success'
        final_results['detection_info'] = detection
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        if final_results['best_match']:
            best = final_results['best_match']
            print(f"\nBest Match:")
            print(f"  Make: {best['make']}")
            print(f"  Model: {best['model']}")
            print(f"  Year Range: {best['year_range']}")
            print(f"  Confidence: {best['confidence']:.3f}")
            
            if final_results['alternatives']:
                print(f"\nAlternatives:")
                for i, alt in enumerate(final_results['alternatives'][:3], 1):
                    print(f"  {i}. {alt['make']} {alt['model']} "
                          f"({alt['year_range']}) - {alt['confidence']:.3f}")
        
        print("\n" + "="*60)
        
        return final_results
    
    def batch_identify(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """
        Identify vehicles in multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of identification results
        """
        results = []
        for i, image in enumerate(images):
            print(f"\n\nProcessing image {i+1}/{len(images)}...")
            result = self.identify(image)
            results.append(result)
        return results
