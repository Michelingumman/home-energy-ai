from src.rl.custom_env import HomeEnergyEnv
from src.rl.config import get_config_dict
import numpy as np
import pandas as pd

# Create environment
config = get_config_dict()
config["debug_prints"] = True  # Ensure debug prints are on to see raw components if needed
config["simulation_days"] = 100 # حوالي 2880 خطوة إذا كانت time_step_minutes = 15
env = HomeEnergyEnv(config)

# Reset environment
obs, info = env.reset()

all_reward_components_over_episode = []
num_steps_to_run = env.simulation_steps # Use simulation_steps from env

print(f"Running simulation for {num_steps_to_run} steps...")

# Take a few random actions
for i in range(num_steps_to_run):
    action = env.action_space.sample()
    
    # # Force a high grid power value to trigger capacity penalty (optional, for specific testing)
    # if i == 2: # Example: trigger on 3rd step
    #     env.battery.soc = 0.5 
    #     env.fixed_baseload_kw = 10.0 
    
    obs, reward, done, truncated, info = env.step(action)
    
    if "reward_components" in info:
        all_reward_components_over_episode.append(info["reward_components"].copy()) # Store a copy

    # # Optional: Print step-by-step info for quick check
    # print(f"\nStep {i+1}:")
    # print(f"  Action: {action}")
    # print(f"  Grid power: {info.get('grid_power_kw', 'N/A'):.2f} kW")
    # print(f"  Battery SoC: {obs['soc'][0]:.2f}")
    # print(f"  Current Price: {info.get('current_price', 'N/A'):.2f} öre/kWh")
    # print(f"  Reward: {reward:.2f}")
    # print(f"  Reward Components: {info.get('reward_components', {})}")
    
    if done or truncated:
        print(f"Episode finished early at step {i+1}")
        break

print(f"\nFinished simulation. Collected {len(all_reward_components_over_episode)} sets of reward components.")
print("\n--- Reward Component Statistics (Unweighted, Signed for Reward/Penalty) ---")

if not all_reward_components_over_episode:
    print("No reward components collected.")
else:
    # Aggregate all components into a dictionary of lists
    aggregated_components = {}
    for step_components in all_reward_components_over_episode:
        for key, value in step_components.items():
            if key not in aggregated_components:
                aggregated_components[key] = []
            aggregated_components[key].append(value)

    for component_name, values_list in aggregated_components.items():
        if not values_list: # Should not happen if component was present
            print(f"\nComponent: {component_name}")
            print("  No data collected for this component.")
            continue

        values_np = np.array(values_list)
        
        non_zero_count = np.count_nonzero(values_np)
        percentage_non_zero = (non_zero_count / len(values_np)) * 100 if len(values_np) > 0 else 0
        
        print(f"\nComponent: {component_name}")
        print(f"  Min: {np.min(values_np):.4f}")
        print(f"  Max: {np.max(values_np):.4f}")
        print(f"  Mean: {np.mean(values_np):.4f}")
        print(f"  Median: {np.median(values_np):.4f}")
        print(f"  Std Dev: {np.std(values_np):.4f}")
        print(f"  Non-zero occurrences: {non_zero_count}/{len(values_np)} ({percentage_non_zero:.2f}%)")

print("\n--- End of Analysis ---")

# Example: How to access a specific component's raw values
# if 'reward_grid_cost' in aggregated_components:
#     grid_costs = aggregated_components['reward_grid_cost']
#     # print(f"\nSample of raw grid costs (first 10): {grid_costs[:10]}") 