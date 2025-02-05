from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Battery and solar branch
        self.battery_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # Appliances branch
        self.appliance_net = nn.Sequential(
            nn.Linear(observation_space["appliance_states"].shape[0], 32),
            nn.ReLU()
        )
        
        # Time encoding branch
        self.time_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )
        
        # Combined layers
        self.combined_net = nn.Sequential(
            nn.Linear(16+32+16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        battery = self.battery_net(observations["battery_soc"])
        appliances = self.appliance_net(observations["appliance_states"])
        time_enc = self.time_net(observations["time_encoding"])
        return self.combined_net(th.cat([battery, appliances, time_enc], dim=1))

class EnergyAgent(PPO):
    def __init__(self, env, config):
        policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "net_arch": [128, 64]
        }
        
        super().__init__(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            policy_kwargs=policy_kwargs
        )

    def save(self, path):
        super().save(path)

    @classmethod
    def load(cls, path, env):
        return super().load(path, env)