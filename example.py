#!/usr/bin/env python3
"""
Example usage of the vehicle identification pipeline
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vehicle_identification import VehicleIdentificationPipeline
from vehicle_identification.utils import load_image, save_image, visualize_detection


def main():
    parser = argparse.ArgumentParser(description='Vehicle Identification System')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to trained model weights (e.g., models/vehicle_classifier.pth)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Initialize pipeline
    print("Initializing Vehicle Identification Pipeline...")
    pipeline = VehicleIdentificationPipeline(config_path=args.config)
    
    # Load custom weights if provided
    if args.weights:
        if os.path.exists(args.weights):
            import json
            print(f"Loading custom model weights from: {args.weights}")
            
            # Load class names if available
            class_names_path = os.path.join(os.path.dirname(args.weights), 'class_names.json')
            class_names = None
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    class_names = json.load(f)
                print(f"Loaded {len(class_names)} class names")
            
            # Load model with weights
            pipeline.classifier.load_model(weights_path=args.weights, class_names=class_names)
        else:
            print(f"Warning: Weights file not found: {args.weights}")
            print("Continuing with default mock model...")
    
    # Load mock database
    pipeline.load_database()
    
    # Run identification
    print(f"\nProcessing image: {args.image}")
    result = pipeline.identify(args.image)
    
    # Display results
    print("\n" + "="*70)
    print("IDENTIFICATION RESULTS")
    print("="*70)
    
    if result['status'] == 'success' and result['best_match']:
        best = result['best_match']
        print(f"\n✓ Best Match:")
        print(f"  Make:        {best['make']}")
        print(f"  Model:       {best['model']}")
        print(f"  Year Range:  {best['year_range']}")
        print(f"  Confidence:  {best['confidence']:.1%}")
        
        if result.get('component_scores'):
            print(f"\n  Component Scores:")
            for component, score in result['component_scores'].items():
                print(f"    {component:15s}: {score:.3f}")
        
        if result['alternatives']:
            print(f"\n  Alternative Matches:")
            for i, alt in enumerate(result['alternatives'], 1):
                print(f"    {i}. {alt['make']} {alt['model']} "
                      f"({alt['year_range']}) - {alt['confidence']:.1%}")
        
        if args.verbose and result.get('evidence'):
            print(f"\n  Evidence Summary:")
            for source, evidence_list in result['evidence'].items():
                if evidence_list:
                    print(f"    {source}: {len(evidence_list)} evidence(s)")
    
    elif result['status'] == 'no_vehicle_detected':
        print(f"\n✗ No vehicle detected in the image")
    else:
        print(f"\n✗ Error: {result.get('message', 'Unknown error')}")
    
    print("="*70)
    
    # Save visualization if requested
    if args.output and result['status'] == 'success':
        print(f"\nSaving visualization to: {args.output}")
        try:
            img = load_image(args.image)
            if result.get('detection_info'):
                det = result['detection_info']
                best = result['best_match']
                label = f"{best['make']} {best['model']}"
                img_vis = visualize_detection(
                    img, det['bbox'], label, best['confidence']
                )
                save_image(img_vis, args.output)
                print("✓ Visualization saved successfully")
        except Exception as e:
            print(f"✗ Error saving visualization: {e}")


if __name__ == '__main__':
    main()
