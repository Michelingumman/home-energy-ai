"""
Simple test script to verify agent loading
"""
import os
import sys

# Add the project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rl.config import get_config_dict
from src.rl.custom_env import HomeEnergyEnv
from src.rl.evaluate_agent import load_agent

def main():
    try:
        # Get configuration
        config = get_config_dict()
        
        # Set up simple test configuration
        config["simulation_days"] = 28
        config["force_specific_start_month"] = True
        config["start_year"] = 2025
        config["start_month"] = 2
        
        # Create environment
        env = HomeEnergyEnv(config=config)
        
        # Try different model paths
        model_paths = [
            os.path.join(config['model_dir'], "short_term_agent_final.zip"),
            os.path.join(config['model_dir'], "best_short_term_model/best_model.zip"),
            os.path.join(config['model_dir'], "optimized_model_20250520_062058.zip"),
        ]
        
        success = False
        
        for model_path in model_paths:
            print(f"\nTrying to load model from: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            
            try:
                agent = load_agent(model_path, env, config)
                if agent is not None:
                    print(f"SUCCESS: Loaded agent from {model_path}")
                    success = True
                    break
                else:
                    print(f"Failed to load model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
        
        if not success:
            print("\nCould not load any model successfully.")
            return 1
        
        print("\nAgent loaded successfully!")
        return 0
    
    except Exception as e:
        import traceback
        print(f"ERROR: An exception occurred:")
        print(f"{str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 