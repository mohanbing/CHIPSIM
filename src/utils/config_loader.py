#!/usr/bin/env python3

import os
import yaml
from typing import Dict, Any

def load_config(config_file: str = "configs/experiments/config_1.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to config file, defaults to "configs/experiments/config_1.yaml" 
                    Can be:
                    - Relative path from project root (e.g., "configs/experiments/config_2.yaml")
                    - Just experiment name (e.g., "config_2" - will look in configs/experiments/)
                    - Absolute path
        
    Returns:
        Dictionary containing configuration values
    """
    # Handle experiment name shortcut (e.g., "config_2" -> "configs/experiments/config_2.yaml")
    if not config_file.endswith('.yaml') and not os.path.isabs(config_file):
        config_file = f"configs/experiments/{config_file}.yaml"
    
    # If relative path, make it relative to the project root (where run_simulation.py is)
    if not os.path.isabs(config_file):
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(script_dir, config_file)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
