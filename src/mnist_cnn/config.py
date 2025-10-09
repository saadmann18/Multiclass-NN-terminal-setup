"""
Configuration management for MNIST CNN project.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for MNIST CNN project."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Default config path relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "default.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Expand user paths
        if "data" in config and "data_dir" in config["data"]:
            config["data"]["data_dir"] = os.path.expanduser(config["data"]["data_dir"])

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'training.epochs')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'training.epochs')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path

        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.

        Args:
            updates: Dictionary of configuration updates
        """

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self._config, updates)

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get("training", {})

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get("data", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get("evaluation", {})

    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get("paths", {})

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
