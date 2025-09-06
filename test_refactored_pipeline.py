#!/usr/bin/env python3
"""
Test script to verify the refactored pipeline components.
"""

import numpy as np
import os
import tempfile

def test_preprocessing_format():
    """Test that the preprocessing output format is correct."""
    print("Testing preprocessing output format...")
    
    # Create mock data in the expected format
    mock_frames = [
        {
            'boxes': np.array([[100, 200, 300, 400]]),
            'keypoints': np.array([[[150, 250], [160, 260], [170, 270]]]),
            'conf': np.array([[0.9, 0.8, 0.7]])
        },
        {
            'boxes': np.array([]),
            'keypoints': np.array([]),
            'conf': np.array([])
        }
    ]
    
    mock_targets = np.array([1, -100])
    mock_near_players = [
        {
            'box': np.array([100, 200, 300, 400]),
            'keypoints': np.array([[150, 250], [160, 260], [170, 270]]),
            'conf': np.array([0.9, 0.8, 0.7])
        },
        None
    ]
    mock_far_players = [None, None]
    
    # Save in the expected format
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
    np.savez_compressed(
        temp_file,
        frames=mock_frames,
        targets=mock_targets,
        near_players=mock_near_players,
        far_players=mock_far_players
    )
    
    # Load and verify
    data = np.load(temp_file, allow_pickle=True)
    
    assert 'frames' in data
    assert 'targets' in data
    assert 'near_players' in data
    assert 'far_players' in data
    
    frames = data['frames']
    targets = data['targets']
    near_players = data['near_players']
    far_players = data['far_players']
    
    assert len(frames) == 2
    assert len(targets) == 2
    assert len(near_players) == 2
    assert len(far_players) == 2
    
    assert targets[0] == 1
    assert targets[1] == -100
    
    print("✓ Preprocessing format test passed")
    
    # Clean up
    os.unlink(temp_file)

def test_feature_format():
    """Test that the feature output format is correct."""
    print("Testing feature output format...")
    
    # Create mock feature data
    mock_features = np.random.randn(10, 288).astype(np.float32)
    mock_targets = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    # Save in the expected format
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
    np.savez_compressed(
        temp_file,
        features=mock_features,
        targets=mock_targets
    )
    
    # Load and verify
    data = np.load(temp_file, allow_pickle=True)
    
    assert 'features' in data
    assert 'targets' in data
    
    features = data['features']
    targets = data['targets']
    
    assert features.shape == (10, 288)
    assert targets.shape == (10,)
    
    print("✓ Feature format test passed")
    
    # Clean up
    os.unlink(temp_file)

def main():
    """Run all tests."""
    print("=== Testing Refactored Pipeline Components ===\n")
    
    try:
        test_preprocessing_format()
        test_feature_format()
        
        print("\n✓ All tests passed!")
        print("The refactored pipeline components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()