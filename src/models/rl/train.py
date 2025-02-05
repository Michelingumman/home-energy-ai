# train.py
from stable_baselines3 import PPO
from custom_env import HomeEnergyEnv
import json

def train(config_path):
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create environment
    env = HomeEnergyEnv(config)
    
    # Initialize SB3 agent
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": CustomFeatureExtractor,
            "net_arch": [256, 128]
        },
        verbose=1
    )
    
    # Train
    model.learn(total_timesteps=config['training_steps'])
    
    # Save
    model.save("energy_manager")