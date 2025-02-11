from stable_baselines3 import PPO
from simulations.custom_env import HomeEnergyEnv
import json

def load_config(path):
    with open(path) as f:
        return json.load(f)

def run_simulation(model_path):
    config = load_config("src/models/rl/rl_config.json")
    demand_model = tf.keras.models.load_model(config["paths"]["demand_model"])
    price_model = tf.keras.models.load_model(config["paths"]["price_model"])
    
    env = HomeEnergyEnv(demand_model, price_model, config)
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        
        if done:
            break

if __name__ == "__main__":
    run_simulation("simulations/results/final_model.zip")