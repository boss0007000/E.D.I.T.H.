"""
Modules package
"""

from .detector import VehicleDetector
from .classifier import VehicleClassifier
from .logo_detector import LogoDetector
from .ocr import VehicleOCR
from .embedding import EmbeddingRetrieval
from .fusion import ResultFusion

__all__ = [
    'VehicleDetector',
    'VehicleClassifier',
    'LogoDetector',
    'VehicleOCR',
    'EmbeddingRetrieval',
    'ResultFusion'
]
