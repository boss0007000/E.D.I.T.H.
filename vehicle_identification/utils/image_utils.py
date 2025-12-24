"""
Utility functions for vehicle identification
"""
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB)
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Output file path
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(output_path)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image
        size: Target size (width, height)
        
    Returns:
        Resized image
    """
    img = Image.fromarray(image)
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def visualize_detection(image: np.ndarray, bbox: list, label: str = "", 
                       confidence: float = 0.0, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Visualize detection on image
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        confidence: Confidence score
        color: Box color (RGB)
        
    Returns:
        Image with visualization
    """
    img = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    if label:
        text = f"{label}"
        if confidence > 0:
            text += f" {confidence:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(img, text, (x1, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def create_comparison_image(images: list, labels: list = None) -> np.ndarray:
    """
    Create side-by-side comparison of images
    
    Args:
        images: List of images
        labels: Optional labels for each image
        
    Returns:
        Combined image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize all to same height
    target_height = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    
    # Concatenate horizontally
    combined = np.concatenate(resized, axis=1)
    
    # Add labels if provided
    if labels:
        for i, label in enumerate(labels):
            x_offset = sum(img.shape[1] for img in resized[:i]) + 10
            cv2.putText(combined, label, (x_offset, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return combined


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted string
    """
    return f"{confidence * 100:.1f}%"
