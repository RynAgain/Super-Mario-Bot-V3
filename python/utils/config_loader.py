"""
Configuration Loader for Super Mario Bros AI Training

This module provides utilities for loading and validating configuration files
from YAML sources with comprehensive error handling and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


class ConfigLoader:
    """
    Loads and validates configuration files for the Mario AI training system.
    
    Supports loading multiple YAML configuration files and merging them into
    a single configuration dictionary with validation and error handling.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files and merge them.
        
        Returns:
            Merged configuration dictionary
        """
        config_files = [
            "training_config.yaml",
            "network_config.yaml", 
            "game_config.yaml",
            "logging_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            self.config.update(file_config)
                            self.logger.info(f"Loaded configuration: {config_file}")
                        else:
                            self.logger.warning(f"Empty configuration file: {config_file}")
                except yaml.YAMLError as e:
                    self.logger.error(f"YAML parsing error in {config_file}: {e}")
                    raise
                except Exception as e:
                    self.logger.error(f"Error loading {config_file}: {e}")
                    raise
            else:
                self.logger.warning(f"Configuration file not found: {config_path}")
        
        # Validate loaded configuration
        self.validate_config()
        
        # Add runtime information
        self._add_runtime_info()
        
        return self.config
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            filename: Name of the configuration file
            
        Returns:
            Configuration dictionary from the file
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration file: {filename}")
                return config or {}
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {filename}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            raise
    
    def validate_config(self):
        """Validate the loaded configuration."""
        self.logger.info("Validating configuration...")
        
        # Validate training configuration
        self._validate_training_config()
        
        # Validate network configuration
        self._validate_network_config()
        
        # Validate enhanced features configuration
        self._validate_enhanced_features_config()
        
        # Validate enhanced communication configuration
        self._validate_enhanced_communication_config()
        
        # Validate game configuration
        self._validate_game_config()
        
        # Validate logging configuration
        self._validate_logging_config()
        
        self.logger.info("Configuration validation completed successfully")
    
    def _validate_training_config(self):
        """Validate training configuration parameters."""
        training_config = self.config.get('training', {})
        
        # Required parameters
        required_params = [
            'learning_rate', 'batch_size', 'gamma', 'epsilon_start', 'epsilon_end'
        ]
        
        for param in required_params:
            if param not in training_config:
                raise ValueError(f"Missing required training parameter: {param}")
        
        # Validate ranges
        if not (0 < training_config['learning_rate'] <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        
        if not (0 <= training_config['gamma'] <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        
        if not (0 <= training_config['epsilon_end'] <= training_config['epsilon_start'] <= 1):
            raise ValueError("Epsilon values must satisfy: 0 <= epsilon_end <= epsilon_start <= 1")
        
        if training_config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
    
    def _validate_network_config(self):
        """Validate network configuration parameters."""
        network_config = self.config.get('network', {})
        
        # Required parameters
        required_params = [
            'frame_stack_size', 'frame_size', 'state_vector_size', 'num_actions'
        ]
        
        for param in required_params:
            if param not in network_config:
                raise ValueError(f"Missing required network parameter: {param}")
        
        # Validate values
        if network_config['frame_stack_size'] <= 0:
            raise ValueError("Frame stack size must be positive")
        
        if network_config['num_actions'] <= 0:
            raise ValueError("Number of actions must be positive")
        
        if len(network_config['frame_size']) != 2:
            raise ValueError("Frame size must be a 2-element list [height, width]")
        
        # Validate enhanced features configuration if present
        enhanced_features = network_config.get('enhanced_features', False)
        if enhanced_features:
            state_vector_size = network_config.get('state_vector_size', 12)
            if state_vector_size not in [12, 20]:
                raise ValueError("State vector size must be 12 (basic) or 20 (enhanced)")
    
    def _validate_enhanced_features_config(self):
        """Validate enhanced features configuration parameters."""
        enhanced_config = self.config.get('enhanced_features', {})
        
        if not enhanced_config:
            return  # Enhanced features not configured
        
        # Validate feature categories
        categories = enhanced_config.get('categories', {})
        if categories:
            expected_categories = [
                'enemy_threat_assessment', 'powerup_detection',
                'environmental_awareness', 'enhanced_mario_state'
            ]
            
            for category in expected_categories:
                if category in categories:
                    feature_count = categories[category]
                    if not isinstance(feature_count, int) or feature_count < 0:
                        raise ValueError(f"Feature count for {category} must be a non-negative integer")
        
        # Validate architecture adjustments
        arch_adjustments = enhanced_config.get('architecture_adjustments', {})
        if arch_adjustments:
            fusion_size = arch_adjustments.get('fusion_hidden_size_20')
            if fusion_size is not None and (not isinstance(fusion_size, int) or fusion_size <= 0):
                raise ValueError("Fusion hidden size must be a positive integer")
        
        # Validate validation settings
        validation_settings = enhanced_config.get('validation', {})
        if validation_settings:
            for setting in ['check_payload_size', 'check_feature_ranges', 'log_feature_stats']:
                if setting in validation_settings and not isinstance(validation_settings[setting], bool):
                    raise ValueError(f"Validation setting {setting} must be a boolean")
    
    def _validate_enhanced_communication_config(self):
        """Validate enhanced communication configuration parameters."""
        comm_config = self.config.get('enhanced_communication', {})
        
        if not comm_config:
            return  # Enhanced communication not configured
        
        # Validate protocol version
        protocol_version = comm_config.get('protocol_version')
        if protocol_version and not isinstance(protocol_version, str):
            raise ValueError("Protocol version must be a string")
        
        # Validate payload size
        payload_size = comm_config.get('payload_size')
        if payload_size is not None:
            if not isinstance(payload_size, int) or payload_size <= 0:
                raise ValueError("Payload size must be a positive integer")
            if payload_size not in [80, 128]:  # Common payload sizes
                self.logger.warning(f"Unusual payload size: {payload_size} bytes")
        
        # Validate binary parsing settings
        binary_parsing = comm_config.get('binary_parsing', {})
        if binary_parsing:
            bool_settings = ['enabled', 'validation_enabled', 'strict_mode']
            for setting in bool_settings:
                if setting in binary_parsing and not isinstance(binary_parsing[setting], bool):
                    raise ValueError(f"Binary parsing setting {setting} must be a boolean")
        
        # Validate reward calculation settings
        reward_calc = comm_config.get('reward_calculation', {})
        if reward_calc:
            if 'enabled' in reward_calc and not isinstance(reward_calc['enabled'], bool):
                raise ValueError("Reward calculation enabled setting must be a boolean")
            
            # Validate feature weights
            feature_weights = reward_calc.get('feature_weights', {})
            if feature_weights:
                for weight_name, weight_value in feature_weights.items():
                    if not isinstance(weight_value, (int, float)) or weight_value < 0:
                        raise ValueError(f"Feature weight {weight_name} must be a non-negative number")
        
        # Validate error handling settings
        error_handling = comm_config.get('error_handling', {})
        if error_handling:
            bool_settings = ['drop_invalid_frames', 'log_validation_warnings', 'log_parsing_errors']
            for setting in bool_settings:
                if setting in error_handling and not isinstance(error_handling[setting], bool):
                    raise ValueError(f"Error handling setting {setting} must be a boolean")
            
            max_errors = error_handling.get('max_consecutive_errors')
            if max_errors is not None and (not isinstance(max_errors, int) or max_errors <= 0):
                raise ValueError("Max consecutive errors must be a positive integer")
        
        # Validate monitoring settings
        monitoring = comm_config.get('monitoring', {})
        if monitoring:
            bool_settings = ['track_statistics', 'log_performance_metrics', 'reset_stats_on_episode']
            for setting in bool_settings:
                if setting in monitoring and not isinstance(monitoring[setting], bool):
                    raise ValueError(f"Monitoring setting {setting} must be a boolean")
    
    def _validate_game_config(self):
        """Validate game configuration parameters."""
        game_config = self.config.get('game', {})
        
        # Validate action space
        actions = game_config.get('actions', {})
        if not actions:
            raise ValueError("Action space configuration is missing")
        
        # Check that action IDs are sequential starting from 0
        action_ids = sorted([int(k) for k in actions.keys()])
        expected_ids = list(range(len(action_ids)))
        
        if action_ids != expected_ids:
            raise ValueError("Action IDs must be sequential starting from 0")
    
    def _validate_logging_config(self):
        """Validate logging configuration parameters."""
        logging_config = self.config.get('logging', {})
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = logging_config.get('level', 'INFO')
        
        if log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
    
    def _add_runtime_info(self):
        """Add runtime information to configuration."""
        self.config['runtime'] = {
            'config_loaded_at': datetime.now().isoformat(),
            'config_dir': str(self.config_dir.absolute()),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration or a specific section.
        
        Args:
            section: Optional section name to retrieve
            
        Returns:
            Configuration dictionary or section
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def save_config(self, output_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration
            config: Configuration to save (uses loaded config if None)
        """
        config_to_save = config or self.config
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {output_path}: {e}")
            raise
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            if config:
                merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load all configuration files.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Merged configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    return loader.load_all_configs()


if __name__ == "__main__":
    # Test configuration loading
    try:
        print("Testing configuration loader...")
        
        # Load all configurations
        config_loader = ConfigLoader()
        config = config_loader.load_all_configs()
        
        print(f"Loaded configuration sections: {list(config.keys())}")
        
        # Test specific section access
        training_config = config_loader.get_config('training')
        print(f"Training config keys: {list(training_config.keys()) if training_config else 'None'}")
        
        network_config = config_loader.get_config('network')
        print(f"Network config keys: {list(network_config.keys()) if network_config else 'None'}")
        
        # Test saving configuration
        config_loader.save_config("test_config_output.yaml")
        print("Configuration saved successfully")
        
        # Cleanup test file
        if os.path.exists("test_config_output.yaml"):
            os.remove("test_config_output.yaml")
        
        print("Configuration loader tests completed successfully!")
        
    except Exception as e:
        print(f"Configuration loader test failed: {e}")
        import traceback
        traceback.print_exc()