#!/usr/bin/env python3
"""
Test reward component scaling and balance after improvements.
"""

import sys
import os
import pathlib as Path

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.rl.config import get_config_dict
from src.rl.custom_env import HomeEnergyEnv
import numpy as np
import logging

def test_reward_components():
    """Test that reward components are balanced and in appropriate ranges."""
    
    # Set up logging to reduce verbosity
    logging.basicConfig(level=logging.WARNING)
    
    print("Testing Reward Component Balance...")
    print("="*60)
    
    # Get the improved config
    config = get_config_dict()
    
    # Test environment
    env = HomeEnergyEnv(config)
    obs, info = env.reset()
    
    # Run for 100 steps to gather statistics
    reward_components = {
        'grid_cost': [],
        'capacity_penalty': [],
        'degradation_cost': [],
        'soc_reward': [],
        'shaping_reward': [],
        'arbitrage_bonus': [],
        'export_bonus': [],
        'night_charging': [],
        'action_penalty': [],
        'solar_soc': [],
        'night_peak_chain': []
    }
    
    total_rewards = []
    action_modified_count = 0
    
    for i in range(100):
        # Test with diverse actions
        if i < 33:
            action = np.array([np.random.uniform(-1, -0.5)])  # Charging
        elif i < 66:
            action = np.array([np.random.uniform(0.5, 1)])    # Discharging  
        else:
            action = np.array([np.random.uniform(-0.3, 0.3)]) # Mild actions
            
        obs, reward, done, truncated, info = env.step(action)
        total_rewards.append(reward)
        
        # Track reward components
        if 'reward_components' in info:
            comp = info['reward_components']
            reward_components['grid_cost'].append(comp.get('reward_grid_cost', 0))
            reward_components['capacity_penalty'].append(comp.get('reward_capacity_penalty', 0))
            reward_components['degradation_cost'].append(comp.get('reward_battery_cost', 0))
            reward_components['soc_reward'].append(comp.get('reward_soc_reward', 0))
            reward_components['shaping_reward'].append(comp.get('reward_shaping', 0))
            reward_components['arbitrage_bonus'].append(comp.get('reward_arbitrage_bonus', 0))
            reward_components['export_bonus'].append(comp.get('reward_export_bonus', 0))
            reward_components['night_charging'].append(comp.get('reward_night_charging', 0))
            reward_components['action_penalty'].append(comp.get('reward_action_mod_penalty', 0))
            reward_components['solar_soc'].append(comp.get('reward_solar_soc', 0))
            reward_components['night_peak_chain'].append(comp.get('reward_night_peak_chain', 0))
            
        # Track action modifications
        if 'action_modified' in info and info['action_modified']:
            action_modified_count += 1
        
        if done or truncated:
            break
    
    env.close()
    
    # Analyze results
    print("Reward Component Statistics (100 steps):")
    print("-" * 60)
    
    for component, values in reward_components.items():
        if values:
            mean_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_val = np.std(values)
            non_zero = sum(1 for v in values if abs(v) > 1e-6)
            
            print(f"{component:15s}: Mean={mean_val:6.2f}, Min={min_val:6.2f}, Max={max_val:6.2f}, "
                  f"Std={std_val:5.2f}, Active={non_zero:2d}/100")
    
    print("-" * 60)
    print(f"Total Reward Stats:")
    print(f"  Mean: {np.mean(total_rewards):.3f}")
    print(f"  Min:  {np.min(total_rewards):.3f}")
    print(f"  Max:  {np.max(total_rewards):.3f}")
    print(f"  Std:  {np.std(total_rewards):.3f}")
    print()
    print(f"Action Modifications: {action_modified_count}/100 ({action_modified_count}%)")
    
    # Check if components are in good ranges
    all_values = []
    for values in reward_components.values():
        all_values.extend([abs(v) for v in values if abs(v) > 1e-6])
    
    if all_values:
        max_abs_component = max(all_values)
        print(f"Largest component magnitude: {max_abs_component:.2f}")
        
        if max_abs_component < 10:
            print("✅ Component scaling looks good (all < 10)")
        elif max_abs_component < 50:
            print("⚠️  Component scaling acceptable but could be improved")
        else:
            print("❌ Component scaling needs improvement")
    
    # Check reward balance
    mean_reward = np.mean(total_rewards)
    if mean_reward > -5:
        print("✅ Mean reward is reasonable")
    else:
        print("❌ Mean reward is too negative")
        
    return True

if __name__ == "__main__":
    success = test_reward_components()
    if success:
        print("\n" + "="*60)
        print("✅ Reward system testing completed!")
    else:
        print("\n" + "="*60)
        print("❌ Issues found in reward system!") 