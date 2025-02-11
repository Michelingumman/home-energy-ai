# train.py
from stable_baselines3 import PPO
from custom_env import HomeEnergyEnv

def train(config):
    env = HomeEnergyEnv(config)
    
    # Custom policy network
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cuda'  # Use GPU if available
    )
    
    model.learn(total_timesteps=1_000_000)
    model.save("energy_manager")