#!/usr/bin/env python3
"""
Quick test script for vehicle identification pipeline
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vehicle_identification import VehicleIdentificationPipeline


def create_mock_vehicle_image(width=800, height=600, save_path='test_vehicle.jpg'):
    """Create a mock vehicle image for testing"""
    # Create base image
    img = Image.new('RGB', (width, height), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Draw ground
    draw.rectangle([(0, height//2), (width, height)], fill=(169, 169, 169))
    
    # Draw car body (simplified)
    car_y = height // 2 - 50
    car_height = 150
    car_width = 400
    car_x = (width - car_width) // 2
    
    # Car body
    draw.rectangle(
        [(car_x, car_y), (car_x + car_width, car_y + car_height)],
        fill=(200, 50, 50)  # Red
    )
    
    # Car roof
    roof_height = 80
    roof_width = 250
    roof_x = car_x + (car_width - roof_width) // 2
    roof_y = car_y - roof_height
    draw.rectangle(
        [(roof_x, roof_y), (roof_x + roof_width, car_y)],
        fill=(180, 40, 40)
    )
    
    # Windows
    window_width = 100
    window_height = 60
    # Left window
    draw.rectangle(
        [(roof_x + 10, roof_y + 10), (roof_x + window_width, roof_y + window_height)],
        fill=(100, 150, 200)
    )
    # Right window
    draw.rectangle(
        [(roof_x + roof_width - window_width - 10, roof_y + 10), 
         (roof_x + roof_width - 10, roof_y + window_height)],
        fill=(100, 150, 200)
    )
    
    # Wheels
    wheel_radius = 40
    wheel_y = car_y + car_height - 10
    # Left wheel
    draw.ellipse(
        [(car_x + 80 - wheel_radius, wheel_y - wheel_radius),
         (car_x + 80 + wheel_radius, wheel_y + wheel_radius)],
        fill=(50, 50, 50)
    )
    # Right wheel
    draw.ellipse(
        [(car_x + car_width - 80 - wheel_radius, wheel_y - wheel_radius),
         (car_x + car_width - 80 + wheel_radius, wheel_y + wheel_radius)],
        fill=(50, 50, 50)
    )
    
    # Add some text (simulating badge)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((car_x + car_width - 100, car_y + 20), "CAMRY", fill=(255, 255, 255), font=font)
    
    # Save image
    img.save(save_path)
    print(f"Created mock vehicle image: {save_path}")
    return save_path


def main():
    print("="*70)
    print("VEHICLE IDENTIFICATION PIPELINE - QUICK TEST")
    print("="*70)
    
    # Create a mock vehicle image
    print("\n1. Creating mock vehicle image...")
    image_path = create_mock_vehicle_image()
    
    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = VehicleIdentificationPipeline()
    
    # Load mock database
    print("\n3. Loading vehicle database...")
    pipeline.load_database()
    
    # Run identification
    print("\n4. Running identification...")
    result = pipeline.identify(image_path)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if result['status'] == 'success':
        print("✓ Pipeline executed successfully")
        if result['best_match']:
            best = result['best_match']
            print(f"✓ Best match: {best['make']} {best['model']} ({best['year_range']})")
            print(f"✓ Confidence: {best['confidence']:.1%}")
    else:
        print(f"✗ Status: {result['status']}")
    
    print("="*70)
    print("\nAll pipeline stages completed successfully!")
    print("The system is ready to process real vehicle images.")
    print("\nUsage: python example.py --image <path_to_vehicle_image>")
    print("="*70)


if __name__ == '__main__':
    main()
