"""
Logo Detection Module
Detects and recognizes vehicle brand logos
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
import cv2


class LogoDetector:
    """Detects and recognizes vehicle logos"""
    
    def __init__(self, model_name: str = "yolov8n",
                 confidence_threshold: float = 0.25,
                 device: str = "cpu"):
        """
        Initialize logo detector
        
        Args:
            model_name: Model name for logo detection
            confidence_threshold: Minimum confidence threshold
            device: Device to run model on
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.logo_classes = self._get_logo_classes()
    
    def _get_logo_classes(self) -> Dict[int, str]:
        """Get mapping of logo class IDs to brand names"""
        return {
            0: 'Toyota',
            1: 'Honda',
            2: 'Ford',
            3: 'Chevrolet',
            4: 'BMW',
            5: 'Mercedes-Benz',
            6: 'Audi',
            7: 'Volkswagen',
            8: 'Nissan',
            9: 'Hyundai',
            10: 'Tesla',
            11: 'Mazda',
            12: 'Subaru',
            13: 'Kia',
            14: 'Lexus'
        }
    
    def load_model(self, weights_path: str = None):
        """
        Load logo detection model
        
        Args:
            weights_path: Path to custom weights (optional)
        """
        try:
            from ultralytics import YOLO
            if weights_path:
                self.model = YOLO(weights_path)
            else:
                # Would use a custom trained model for logos
                # For demo, using None to trigger mock
                self.model = None
            print("Logo detector initialized")
        except Exception as e:
            print(f"Error loading logo detector: {e}")
            self.model = None
    
    def detect_logos(self, image: np.ndarray) -> List[Dict]:
        """
        Detect logos in the vehicle image
        
        Args:
            image: Vehicle image (typically cropped to vehicle)
            
        Returns:
            List of detected logos with brand and confidence
        """
        if self.model is None:
            return self._mock_detect_logo(image)
        
        try:
            results = self.model(image, conf=self.confidence_threshold, device=self.device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    brand = self.logo_classes.get(cls, f"Brand_{cls}")
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'brand': brand,
                        'confidence': conf
                    })
            
            return detections
        except Exception as e:
            print(f"Error during logo detection: {e}")
            return self._mock_detect_logo(image)
    
    def _mock_detect_logo(self, image: np.ndarray) -> List[Dict]:
        """Mock logo detection for demonstration"""
        h, w = image.shape[:2]
        
        # Assume logo is in top-center region
        x1 = int(w * 0.4)
        y1 = int(h * 0.1)
        x2 = int(w * 0.6)
        y2 = int(h * 0.25)
        
        return [{
            'bbox': [x1, y1, x2, y2],
            'brand': 'Toyota',
            'confidence': 0.75
        }]
    
    def extract_logo_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract logo region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped logo image
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def get_top_brand(self, detections: List[Dict]) -> str:
        """
        Get the most confident brand from detections
        
        Args:
            detections: List of logo detections
            
        Returns:
            Brand name with highest confidence
        """
        if not detections:
            return "Unknown"
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        return best_detection['brand']
