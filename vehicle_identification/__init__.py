"""
Vehicle Identification Package
Complete pipeline for vehicle identification from images
"""

from .pipeline import VehicleIdentificationPipeline
from .modules.detector import VehicleDetector
from .modules.classifier import VehicleClassifier
from .modules.logo_detector import LogoDetector
from .modules.ocr import VehicleOCR
from .modules.embedding import EmbeddingRetrieval
from .modules.fusion import ResultFusion

__version__ = "1.0.0"
__all__ = [
    'VehicleIdentificationPipeline',
    'VehicleDetector',
    'VehicleClassifier',
    'LogoDetector',
    'VehicleOCR',
    'EmbeddingRetrieval',
    'ResultFusion'
]
