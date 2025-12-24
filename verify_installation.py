#!/usr/bin/env python3
"""
Verification script to check all imports and module structure
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    try:
        # Main package
        import vehicle_identification
        print("✓ vehicle_identification")
        
        # Pipeline
        from vehicle_identification import VehicleIdentificationPipeline
        print("✓ VehicleIdentificationPipeline")
        
        # Individual modules
        from vehicle_identification.modules import VehicleDetector
        print("✓ VehicleDetector")
        
        from vehicle_identification.modules import VehicleClassifier
        print("✓ VehicleClassifier")
        
        from vehicle_identification.modules import LogoDetector
        print("✓ LogoDetector")
        
        from vehicle_identification.modules import VehicleOCR
        print("✓ VehicleOCR")
        
        from vehicle_identification.modules import EmbeddingRetrieval
        print("✓ EmbeddingRetrieval")
        
        from vehicle_identification.modules import ResultFusion
        print("✓ ResultFusion")
        
        # Utils
        from vehicle_identification.utils import load_image, save_image
        print("✓ Utilities")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import error: {e}")
        return False


def test_initialization():
    """Test basic initialization"""
    print("\nTesting module initialization...")
    
    try:
        from vehicle_identification import VehicleIdentificationPipeline
        
        # Initialize pipeline
        pipeline = VehicleIdentificationPipeline()
        print("✓ Pipeline initialized")
        
        # Check modules
        assert pipeline.detector is not None, "Detector not initialized"
        print("✓ Detector module")
        
        assert pipeline.classifier is not None, "Classifier not initialized"
        print("✓ Classifier module")
        
        assert pipeline.logo_detector is not None, "Logo detector not initialized"
        print("✓ Logo detector module")
        
        assert pipeline.ocr is not None, "OCR not initialized"
        print("✓ OCR module")
        
        assert pipeline.embedding is not None, "Embedding not initialized"
        print("✓ Embedding module")
        
        assert pipeline.fusion is not None, "Fusion not initialized"
        print("✓ Fusion module")
        
        print("\n✓ All modules initialized successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from vehicle_identification import VehicleIdentificationPipeline
        
        pipeline = VehicleIdentificationPipeline()
        
        # Check config structure
        assert 'detection' in pipeline.config
        assert 'classification' in pipeline.config
        assert 'logo_detection' in pipeline.config
        assert 'ocr' in pipeline.config
        assert 'embedding' in pipeline.config
        assert 'fusion' in pipeline.config
        
        print("✓ Configuration loaded correctly")
        print(f"  - Detection model: {pipeline.config['detection']['model']}")
        print(f"  - Classification model: {pipeline.config['classification']['model_name']}")
        print(f"  - Embedding model: {pipeline.config['embedding']['model_name']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("VEHICLE IDENTIFICATION SYSTEM - VERIFICATION")
    print("="*70)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test initialization
    results.append(("Initialization", test_initialization()))
    
    # Test configuration
    results.append(("Configuration", test_configuration()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All verification tests passed!")
        print("The system is ready to use.")
        return 0
    else:
        print("\n✗ Some verification tests failed.")
        print("Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
