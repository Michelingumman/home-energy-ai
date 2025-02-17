import json
from pathlib import Path

class FeatureConfig:
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.json"
        self.load_config()
    
    def load_config(self):
        """Load the feature configuration from JSON"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set attributes for easy access
        self.feature_groups = self.config["feature_groups"]
        self.metadata = self.config["feature_metadata"]
        self.model_config = self.config["model_config"]
        
        # Set individual feature groups
        self.price_cols = self.feature_groups["price_cols"]
        self.grid_cols = self.feature_groups["grid_cols"]
        self.cyclical_cols = self.feature_groups["cyclical_cols"]
        self.binary_cols = self.feature_groups["binary_cols"]
        
        # Set important metadata
        self.target_feature = self.metadata["target_feature"]
        self.feature_order = self.metadata["feature_order"]
        
        # Set model configuration
        self.architecture = self.model_config["architecture"]
        self.training = self.model_config["training"]
        self.callbacks = self.model_config["callbacks"]
        self.data_split = self.model_config["data_split"]
        self.scaling = self.model_config["scaling"]
    
    @property
    def total_features(self):
        """Calculate total number of features dynamically"""
        return len(self.get_all_features())
    
    def get_all_features(self):
        """Get all features in the correct order"""
        all_features = []
        for group in self.feature_order:
            all_features.extend(self.feature_groups[group])
        return all_features
    
    def get_feature_group(self, group_name):
        """Get features for a specific group"""
        return self.feature_groups.get(group_name, [])
    
    def get_ordered_features(self):
        """Get all features in the training order with target first"""
        features = [self.target_feature]  # Target always first
        features.extend([f for f in self.price_cols if f != self.target_feature])
        
        # Add other features in order
        for group in self.feature_order[1:]:  # Skip price_cols as we handled it
            features.extend(self.feature_groups[group])
        
        return features
    
    def verify_features(self, available_features):
        """Verify that all required features are available"""
        required_features = set(self.get_all_features())
        missing_features = required_features - set(available_features)
        return list(missing_features)
    
    def get_model_architecture(self):
        """Get the model architecture configuration"""
        return self.architecture
    
    def get_training_params(self):
        """Get training parameters"""
        return self.training
    
    def get_callback_params(self):
        """Get callback configurations"""
        return self.callbacks
    
    def get_data_split_ratios(self):
        """Get data split ratios"""
        return self.data_split
    
    def get_scaling_params(self):
        """Get scaling configurations"""
        return self.scaling

# Create a singleton instance
feature_config = FeatureConfig() 