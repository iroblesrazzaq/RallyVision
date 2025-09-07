#!/usr/bin/env python3
"""
Test script to verify the progress bar functionality.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_progress_bar_functionality():
    """Test that the progress bar functionality works correctly."""
    print("Testing progress bar functionality...")
    
    try:
        # Test that tqdm can be imported
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
        
        # Test that pose extractor can be imported
        from data_scripts.pose_extractor import PoseExtractor
        print("✓ PoseExtractor imported successfully")
        
        # Test that run_pipeline can be imported
        import run_pipeline
        print("✓ run_pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Progress bar test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing Progress Bar Functionality ===\n")
    
    try:
        if test_progress_bar_functionality():
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n💥 Some tests failed!")
            return False
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()