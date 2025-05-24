#!/usr/bin/env python3
"""
Test script to verify environment observation shapes after changing 
load forecast from 4*24 to 3*24 hours.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rl.custom_env import HomeEnergyEnv
from src.rl.config import get_config_dict
import numpy as np

def test_observation_shapes():
    """Test that all observation components have the expected shapes."""
    
    print("Testing environment observation shapes...")
    
    # Create environment
    config = get_config_dict()
    config['debug_prints'] = False  # Reduce noise
    config['simulation_days'] = 7   # Shorter test
    
    env = HomeEnergyEnv(config=config)
    
    # Reset environment  
    obs, info = env.reset(seed=42)
    
    print("\nObservation Space Definition:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    
    print("\nActual Observation Shapes:")
    total_size = 0
    for key, value in obs.items():
        shape = value.shape
        size = np.prod(shape)
        total_size += size
        print(f"  {key}: {shape} (size: {size})")
    
    print(f"\nTotal observation size: {total_size}")
    
    # Expected sizes based on your changes:
    expected_sizes = {
        'soc': 1,
        'time_idx': 3, 
        'price_forecast': 24,
        'solar_forecast': 72,  # 3*24
        'capacity_metrics': 5,
        'price_averages': 2,
        'is_night_discount': 1,
        'load_forecast': 72    # Changed from 96 (4*24) to 72 (3*24)
    }
    
    expected_total = sum(expected_sizes.values())
    print(f"Expected total size: {expected_total}")
    
    # Verify each component
    print("\nVerification:")
    all_correct = True
    for key, expected_size in expected_sizes.items():
        actual_size = np.prod(obs[key].shape)
        status = "✓" if actual_size == expected_size else "✗"
        print(f"  {key}: {actual_size} == {expected_size} {status}")
        if actual_size != expected_size:
            all_correct = False
    
    print(f"\nOverall: {'✓ All shapes correct!' if all_correct else '✗ Shape mismatch detected!'}")
    
    env.close()
    return all_correct

if __name__ == "__main__":
    success = test_observation_shapes()
    sys.exit(0 if success else 1) 