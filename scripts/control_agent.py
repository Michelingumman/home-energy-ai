class BatteryController:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # Define reward components
        self.peak_penalty = 10.0
        self.price_reward = 1.0
        self.degradation_penalty = 0.5
        
    def get_action(self, state):
        # State includes: predicted_demand, energy_price, 
        # battery_soc, solar_generation
        action = self.actor(state)
        return self.constrain_action(action)
    
    def compute_reward(self, state, action, next_state):
        peak_cost = self.calculate_peak_penalty(state)
        price_benefit = self.calculate_price_benefit(state, action)
        degradation = self.calculate_degradation(action)
        
        return (price_benefit 
                - self.peak_penalty * peak_cost 
                - self.degradation_penalty * degradation)