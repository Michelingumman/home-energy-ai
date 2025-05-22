"""
Test imports for RecurrentPPO
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sb3_contrib.ppo_recurrent import RecurrentPPO
    print("✅ RecurrentPPO imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import RecurrentPPO: {e}")
    print("You may need to install sb3-contrib with:")
    print("pip install sb3-contrib")
    exit(1)

# Also test our custom agent
try:
    from src.rl.agent import RecurrentEnergyAgent
    print("✅ RecurrentEnergyAgent imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import RecurrentEnergyAgent: {e}")
    exit(1)
    
print("All imports successful! You can now run the RecurrentPPO agent.") 