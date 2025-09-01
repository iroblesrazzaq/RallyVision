#!/usr/bin/env python3
"""
Test script to verify YOLO setup and dependencies
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if YOLO models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Test nano model (should exist)
        if os.path.exists("../yolov8n.pt"):
            print("✓ yolov8n.pt found")
            model = YOLO("../yolov8n.pt")
            print("✓ Nano model loaded successfully")
        else:
            print("✗ yolov8n.pt not found")
            return False
            
        # Test if we can run inference on a dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✓ Model inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_video_access():
    """Test if we can access video files"""
    print("\nTesting video access...")
    
    video_dir = "../raw_videos"
    if not os.path.exists(video_dir):
        print(f"✗ Video directory {video_dir} not found")
        return False
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"✓ Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("✗ No video files found")
        return False
    
    # Test if we can open the first video
    try:
        import cv2
        first_video = os.path.join(video_dir, video_files[0])
        cap = cv2.VideoCapture(first_video)
        if cap.isOpened():
            print(f"✓ Successfully opened video: {video_files[0]}")
            cap.release()
        else:
            print(f"✗ Failed to open video: {video_files[0]}")
            return False
    except Exception as e:
        print(f"✗ Video access test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("YOLO Player Detector - Setup Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_model_loading():
        tests_passed += 1
    
    if test_video_access():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Setup is ready.")
        print("\nYou can now run:")
        print("  python yolo_player_detector.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
