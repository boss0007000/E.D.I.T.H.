"""
Vehicle Make/Model Classification Module
Uses deep learning for vehicle classification
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import timm


class VehicleClassifier:
    """Classifies vehicle make and model"""
    
    def __init__(self, model_name: str = "resnet50",
                 num_classes: int = 100,
                 confidence_threshold: float = 0.1,
                 top_k: int = 5,
                 device: str = "cpu"):
        """
        Initialize the vehicle classifier
        
        Args:
            model_name: Model architecture name
            num_classes: Number of vehicle classes
            confidence_threshold: Minimum confidence threshold
            top_k: Number of top predictions to return
            device: Device to run model on
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = torch.device(device)
        self.model = None
        self.class_names = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, weights_path: str = None, class_names: List[str] = None):
        """
        Load the classification model
        
        Args:
            weights_path: Path to model weights (optional)
            class_names: List of class names
        """
        try:
            # Create model using timm
            self.model = timm.create_model(self.model_name, 
                                          pretrained=True if weights_path is None else False,
                                          num_classes=self.num_classes)
            
            if weights_path:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load or generate class names
            if class_names:
                self.class_names = class_names
            else:
                self.class_names = self._generate_mock_class_names()
            
            print(f"Loaded {self.model_name} classifier successfully")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            print("Using mock classifier for demonstration")
            self.model = None
            self.class_names = self._generate_mock_class_names()
    
    def _generate_mock_class_names(self) -> List[str]:
        """Generate mock class names for demonstration"""
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes-Benz',
                'Audi', 'Volkswagen', 'Nissan', 'Hyundai']
        models = ['Camry', 'Accord', 'F-150', 'Silverado', '3 Series', 'C-Class',
                 'A4', 'Jetta', 'Altima', 'Elantra']
        
        class_names = []
        for make in makes:
            for model in models[:5]:
                class_names.append(f"{make} {model}")
        
        # Pad to num_classes
        while len(class_names) < self.num_classes:
            class_names.append(f"Vehicle_Class_{len(class_names)}")
        
        return class_names[:self.num_classes]
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for classification
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
    
    def classify(self, image: np.ndarray) -> List[Dict]:
        """
        Classify vehicle make and model
        
        Args:
            image: Cropped vehicle image
            
        Returns:
            List of predictions with class, confidence, and details
        """
        if self.model is None:
            return self._mock_classify()
        
        try:
            # Preprocess
            input_tensor = self.preprocess(image).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, self.top_k, dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                if prob >= self.confidence_threshold:
                    class_name = self.class_names[idx]
                    # Parse make and model
                    parts = class_name.split()
                    make = parts[0] if len(parts) > 0 else "Unknown"
                    model = ' '.join(parts[1:]) if len(parts) > 1 else "Unknown"
                    
                    predictions.append({
                        'class_idx': int(idx),
                        'class_name': class_name,
                        'make': make,
                        'model': model,
                        'confidence': float(prob),
                        'year_range': self._estimate_year_range(class_name)
                    })
            
            return predictions
        except Exception as e:
            print(f"Error during classification: {e}")
            return self._mock_classify()
    
    def _mock_classify(self) -> List[Dict]:
        """Mock classification for demonstration"""
        predictions = [
            {
                'class_idx': 0,
                'class_name': 'Toyota Camry',
                'make': 'Toyota',
                'model': 'Camry',
                'confidence': 0.65,
                'year_range': '2018-2023'
            },
            {
                'class_idx': 5,
                'class_name': 'Honda Accord',
                'make': 'Honda',
                'model': 'Accord',
                'confidence': 0.20,
                'year_range': '2017-2023'
            },
            {
                'class_idx': 12,
                'class_name': 'Nissan Altima',
                'make': 'Nissan',
                'model': 'Altima',
                'confidence': 0.10,
                'year_range': '2019-2023'
            }
        ]
        return predictions
    
    def _estimate_year_range(self, class_name: str) -> str:
        """Estimate year range (placeholder - would need actual model)"""
        return "2018-2023"
