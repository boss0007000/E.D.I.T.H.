"""
OCR Module
Extracts text from vehicle images (license plates, badges, etc.)
"""
import numpy as np
from typing import List, Dict, Tuple
import cv2
import re


class VehicleOCR:
    """Performs OCR on vehicle images"""
    
    def __init__(self, reader: str = "easyocr",
                 languages: List[str] = ["en"],
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize OCR reader
        
        Args:
            reader: OCR engine to use ('easyocr' or 'tesseract')
            languages: Languages for OCR
            confidence_threshold: Minimum confidence threshold
            device: Device to run on
        """
        self.reader_type = reader
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.reader = None
    
    def load_reader(self):
        """Load OCR reader"""
        try:
            if self.reader_type == "easyocr":
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=(self.device == "cuda"))
                print("EasyOCR reader loaded successfully")
            elif self.reader_type == "tesseract":
                import pytesseract
                self.reader = pytesseract
                print("Tesseract reader loaded successfully")
        except Exception as e:
            print(f"Error loading OCR reader: {e}")
            print("Using mock OCR for demonstration")
            self.reader = None
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Extract text from image
        
        Args:
            image: Input image
            preprocess: Whether to preprocess image
            
        Returns:
            List of detected text with bounding boxes and confidence
        """
        if self.reader is None:
            return self._mock_ocr()
        
        try:
            if preprocess:
                processed = self.preprocess_for_ocr(image)
            else:
                processed = image
            
            if self.reader_type == "easyocr":
                results = self.reader.readtext(processed)
                
                text_results = []
                for bbox, text, conf in results:
                    if conf >= self.confidence_threshold:
                        text_results.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox
                        })
                
                return text_results
            
            elif self.reader_type == "tesseract":
                text = self.reader.image_to_string(processed)
                data = self.reader.image_to_data(processed, output_type=self.reader.Output.DICT)
                
                text_results = []
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    conf = float(data['conf'][i])
                    if conf >= self.confidence_threshold * 100:  # Tesseract uses 0-100 scale
                        text = data['text'][i].strip()
                        if text:
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            text_results.append({
                                'text': text,
                                'confidence': conf / 100.0,
                                'bbox': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                            })
                
                return text_results
        
        except Exception as e:
            print(f"Error during OCR: {e}")
            return self._mock_ocr()
    
    def _mock_ocr(self) -> List[Dict]:
        """Mock OCR for demonstration"""
        return [
            {
                'text': 'CAMRY',
                'confidence': 0.85,
                'bbox': [[100, 50], [200, 50], [200, 80], [100, 80]]
            },
            {
                'text': 'XLE',
                'confidence': 0.72,
                'bbox': [[220, 50], [280, 50], [280, 80], [220, 80]]
            }
        ]
    
    def extract_model_info(self, text_results: List[Dict]) -> List[str]:
        """
        Extract potential model information from OCR results
        
        Args:
            text_results: OCR results
            
        Returns:
            List of potential model names/badges
        """
        model_info = []
        
        # Common vehicle model patterns
        patterns = [
            r'\b[A-Z]{2,}\b',  # Uppercase abbreviations (XLE, SE, LX)
            r'\b\d{3}[a-zA-Z]?\b',  # Numbers like 350, 450h
            r'\b[A-Z][0-9]+\b',  # Alphanumeric like X5, Q7
        ]
        
        for result in text_results:
            text = result['text'].strip()
            
            # Check patterns
            for pattern in patterns:
                matches = re.findall(pattern, text)
                model_info.extend(matches)
            
            # Also add high-confidence text as-is
            if result['confidence'] > 0.7 and len(text) > 1:
                model_info.append(text)
        
        return list(set(model_info))  # Remove duplicates
    
    def extract_brand_info(self, text_results: List[Dict]) -> List[str]:
        """
        Extract potential brand information from OCR results
        
        Args:
            text_results: OCR results
            
        Returns:
            List of potential brand names
        """
        known_brands = [
            'TOYOTA', 'HONDA', 'FORD', 'CHEVROLET', 'CHEVY', 'BMW', 
            'MERCEDES', 'BENZ', 'AUDI', 'VOLKSWAGEN', 'VW', 'NISSAN',
            'HYUNDAI', 'TESLA', 'MAZDA', 'SUBARU', 'KIA', 'LEXUS'
        ]
        
        brands = []
        for result in text_results:
            text = result['text'].upper().strip()
            if text in known_brands:
                brands.append(text)
        
        return list(set(brands))
