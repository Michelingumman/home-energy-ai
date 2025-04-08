import json
from pathlib import Path
import logging

class FeatureConfig:
    """Configuration class for managing features used in price predictions."""
    
    def __init__(self, config_path=None):
        """
        Initialize the feature configuration.
        
        Args:
            config_path: Optional path to config file. If None, uses default path.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.json"
        else:
            self.config_path = Path(config_path)
            
        self.load_config()
    
    def load_config(self):
        """Load the feature configuration from JSON with error handling."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Set attributes for easy access
            self.feature_groups = self.config.get("feature_groups", {})
            self.metadata = self.config.get("feature_metadata", {})
            self.model_config = self.config.get("model_config", {})
            
            # Success message
            logging.info(f"Successfully loaded configuration from {self.config_path}")
            
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {self.config_path}")
            self._set_default_config()
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file {self.config_path}")
            self._set_default_config()
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            self._set_default_config()
    
    def _set_default_config(self):
        """Set default configuration if the config file cannot be loaded."""
        logging.warning("Using default configuration")
        
        # Default feature groups
        self.feature_groups = {
            "price_cols": ["SE3_price_ore"],
            "grid_cols": [],
            "cyclical_cols": [],
            "binary_cols": []
        }
        
        # Default metadata
        self.metadata = {
            "target_feature": "SE3_price_ore",
            "feature_order": ["price_cols", "cyclical_cols", "binary_cols", "grid_cols"]
        }
        
        # Default model configuration
        self.model_config = {
            "architecture": {},
            "training": {},
            "callbacks": {},
            "data_split": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            "scaling": {},
            "feature_weights": {
                "price_cols": 1.0,
                "cyclical_cols": 1.0,
                "binary_cols": 1.0,
                "grid_cols": 1.0,
                "enable_weighting": False
            }
        }
    
    def get_price_cols(self):
        """Get price-related feature columns."""
        return self.feature_groups.get("price_cols", [])
    
    def get_grid_cols(self):
        """Get grid-related feature columns."""
        return self.feature_groups.get("grid_cols", [])
    
    def get_cyclical_cols(self):
        """Get cyclical feature columns."""
        return self.feature_groups.get("cyclical_cols", [])
    
    def get_binary_cols(self):
        """Get binary feature columns."""
        return self.feature_groups.get("binary_cols", [])
    
    def get_target_name(self):
        """Get the target feature name."""
        return self.metadata.get("target_feature", "SE3_price_ore")
    
    def get_all_feature_names(self):
        """Get all features in the proper order as defined in feature_order."""
        all_features = []
        feature_order = self.metadata.get("feature_order", [])
        
        for group in feature_order:
            if group in self.feature_groups:
                all_features.extend(self.feature_groups[group])
        
        return all_features
    
    def get_ordered_features(self):
        """Get all features with target first, then other features in order."""
        target = self.get_target_name()
        all_features = self.get_all_feature_names()
        
        # If target is in the features, move it to the front
        if target in all_features:
            all_features.remove(target)
            return [target] + all_features
        
        return all_features
    
    def get_feature_weights(self):
        """Get feature weights from config"""
        default_weights = {
            'price_cols': 1.0,
            'cyclical_cols': 1.0,
            'binary_cols': 1.0,
            'grid_cols': 1.0,
            'enable_weighting': False
        }
        
        # Get feature weights from model_config
        feature_weights = self.model_config.get("feature_weights", {})
        
        # Merge with defaults
        weights = default_weights.copy()
        weights.update(feature_weights)
        
        return weights
    
    def missing_columns(self, df):
        """Check for missing required columns in the dataframe."""
        required = set(self.get_all_feature_names())
        available = set(df.columns)
        missing = required - available
        return list(missing) if missing else None
    
    def get_architecture_params(self):
        """Get model architecture parameters."""
        return self.model_config.get("architecture", {})
    
    def get_training_params(self):
        """Get training parameters."""
        return self.model_config.get("training", {})
    
    def get_callback_params(self):
        """Get callback configurations."""
        return self.model_config.get("callbacks", {})
    
    def get_data_split_ratios(self):
        """Get data split ratios."""
        return self.model_config.get("data_split", {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1})
    
    def get_scaling_params(self):
        """Get scaling configurations."""
        return self.model_config.get("scaling", {})
    
    def save_config(self, path=None):
        """Save current configuration to file."""
        save_path = path if path else self.config_path
        
        try:
            # Prepare current config
            config_to_save = {
                "feature_groups": self.feature_groups,
                "feature_metadata": self.metadata,
                "model_config": self.model_config
            }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            
            logging.info(f"Configuration saved to {save_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            return False 