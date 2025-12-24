"""
Vehicle Detection Module
Uses YOLO for detecting vehicles in images
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import cv2


class VehicleDetector:
    """Detects vehicles in images using YOLO"""
    
    def __init__(self, model_name: str = "yolov8n", 
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 device: str = "cpu",
                 target_classes: List[int] = [2, 3, 5, 7]):
        """
        Initialize the vehicle detector
        
        Args:
            model_name: YOLO model name
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run model on
            target_classes: COCO classes for vehicles (2=car, 3=motorcycle, 5=bus, 7=truck)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes
        self.model = None
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"Loaded {self.model_name} successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Using mock detector for demonstration")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in the image
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of detections with bounding boxes, class, and confidence
        """
        if self.model is None:
            # Mock detection for demonstration
            return self._mock_detect(image)
        
        try:
            results = self.model(image, conf=self.confidence_threshold, 
                               iou=self.iou_threshold, device=self.device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': cls,
                            'confidence': conf,
                            'class_name': self._get_class_name(cls)
                        })
            
            return detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return self._mock_detect(image)
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict]:
        """Mock detection for demonstration when model is not available"""
        h, w = image.shape[:2]
        # Assume vehicle is in center 80% of image
        margin = 0.1
        x1 = int(w * margin)
        y1 = int(h * margin)
        x2 = int(w * (1 - margin))
        y2 = int(h * (1 - margin))
        
        return [{
            'bbox': [x1, y1, x2, y2],
            'class': 2,  # car
            'confidence': 0.85,
            'class_name': 'car'
        }]
    
    def _get_class_name(self, cls: int) -> str:
        """Get class name from class ID"""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(cls, 'vehicle')
    
    def crop_vehicle(self, image: np.ndarray, bbox: List[int], 
                     padding: float = 0.1) -> np.ndarray:
        """
        Crop vehicle from image with optional padding
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding ratio to add around bbox
            
        Returns:
            Cropped vehicle image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def detect_and_crop(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        Detect vehicles and crop them
        
        Args:
            image: Input image
            
        Returns:
            List of (cropped_image, detection_info) tuples
        """
        detections = self.detect(image)
        crops = []
        
        for detection in detections:
            cropped = self.crop_vehicle(image, detection['bbox'])
            crops.append((cropped, detection))
        
        return crops
