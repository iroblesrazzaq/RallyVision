#!/usr/bin/env python3
"""
Test script to verify the heuristic processor is working correctly.
"""

import numpy as np
from heuristic_processor import HeuristicProcessor

def test_heuristic_processor():
    """Test the heuristic processor with a simple example."""
    # Initialize the heuristic processor
    processor = HeuristicProcessor(screen_width=1280, screen_height=720)
    
    # Create some test data with two players
    # Player 1 (near player) - bottom half of screen
    player1_box = np.array([100, 500, 200, 700])  # [x1, y1, x2, y2]
    player1_keypoints = np.random.rand(17, 2) * 100  # Random keypoints
    player1_conf = np.random.rand(17)  # Random confidence scores
    
    # Player 2 (far player) - top half of screen
    player2_box = np.array([800, 200, 900, 400])  # [x1, y1, x2, y2]
    player2_keypoints = np.random.rand(17, 2) * 100  # Random keypoints
    player2_conf = np.random.rand(17)  # Random confidence scores
    
    # Create frame data
    frame_data = {
        'boxes': np.array([player1_box, player2_box]),
        'keypoints': np.array([player1_keypoints, player2_keypoints]),
        'conf': np.array([player1_conf, player2_conf])
    }
    
    # Apply the heuristic processor
    assigned_players = processor.assign_players(frame_data)
    
    # Create feature vector
    feature_vector = processor.create_feature_vector(assigned_players)
    
    # Print results
    print("Test Results:")
    print(f"Near player assigned: {assigned_players['near_player'] is not None}")
    print(f"Far player assigned: {assigned_players['far_player'] is not None}")
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector (first 10 elements): {feature_vector[:10]}")

if __name__ == "__main__":
    test_heuristic_processor()