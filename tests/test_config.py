"""
Tests for configuration management.
"""

import os
import tempfile
import yaml
import pytest
from pathlib import Path
from src.mnist_cnn.config import Config


class TestConfig:
    """Test cases for configuration management."""

    def test_config_initialization_with_default(self):
        """Test config initialization with default config file."""
        # This test assumes the default config exists
        config = Config()
        assert config.config_path.name == "default.yaml"
        assert config._config is not None

    def test_config_initialization_with_custom_path(self):
        """Test config initialization with custom config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"model": {"n_channels": 1}, "training": {"epochs": 5}}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            assert config.config_path == Path(temp_path)
            assert config.get("model.n_channels") == 1
            assert config.get("training.epochs") == 5
        finally:
            os.unlink(temp_path)

    def test_config_file_not_found(self):
        """Test config raises error when file not found."""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_config_get_simple_key(self):
        """Test getting simple configuration key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"test_key": "test_value"}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            assert config.get("test_key") == "test_value"
        finally:
            os.unlink(temp_path)

    def test_config_get_nested_key(self):
        """Test getting nested configuration key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"section": {"subsection": {"key": "nested_value"}}}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            assert config.get("section.subsection.key") == "nested_value"
        finally:
            os.unlink(temp_path)

    def test_config_get_default_value(self):
        """Test getting configuration with default value."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"existing_key": "value"}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            assert config.get("nonexistent_key", "default") == "default"
            assert config.get("existing_key", "default") == "value"
        finally:
            os.unlink(temp_path)

    def test_config_set_simple_key(self):
        """Test setting simple configuration key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            config.set("new_key", "new_value")
            assert config.get("new_key") == "new_value"
        finally:
            os.unlink(temp_path)

    def test_config_set_nested_key(self):
        """Test setting nested configuration key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            config.set("section.subsection.key", "nested_value")
            assert config.get("section.subsection.key") == "nested_value"
        finally:
            os.unlink(temp_path)

    def test_config_update(self):
        """Test updating configuration with dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"existing": {"key": "old_value"}, "other": "unchanged"}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            updates = {
                "existing": {"key": "new_value", "new_key": "new"},
                "completely_new": "section",
            }
            config.update(updates)

            assert config.get("existing.key") == "new_value"
            assert config.get("existing.new_key") == "new"
            assert config.get("completely_new") == "section"
            assert config.get("other") == "unchanged"
        finally:
            os.unlink(temp_path)

    def test_config_save(self):
        """Test saving configuration to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"original": "value"}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            config.set("new_key", "new_value")
            config.save()

            # Load the file directly to verify
            with open(temp_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["original"] == "value"
            assert saved_config["new_key"] == "new_value"
        finally:
            os.unlink(temp_path)

    def test_config_save_to_different_path(self):
        """Test saving configuration to different file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"test": "value"}
            yaml.dump(test_config, f)
            temp_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            save_path = f2.name

        try:
            config = Config(temp_path)
            config.set("new_key", "new_value")
            config.save(save_path)

            # Load the save file
            with open(save_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["test"] == "value"
            assert saved_config["new_key"] == "new_value"
        finally:
            os.unlink(temp_path)
            os.unlink(save_path)

    def test_config_property_accessors(self):
        """Test configuration property accessors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {
                "model": {"n_channels": 1},
                "training": {"epochs": 10},
                "data": {"batch_size": 32},
                "evaluation": {"show_plot": True},
                "paths": {"artifacts_dir": "artifacts"},
            }
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)

            assert config.model == {"n_channels": 1}
            assert config.training == {"epochs": 10}
            assert config.data == {"batch_size": 32}
            assert config.evaluation == {"show_plot": True}
            assert config.paths == {"artifacts_dir": "artifacts"}
        finally:
            os.unlink(temp_path)

    def test_config_expand_user_paths(self):
        """Test that user paths are expanded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {"data": {"data_dir": "~/test/path"}}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            data_dir = config.get("data.data_dir")

            # Should be expanded
            assert not data_dir.startswith("~")
            assert data_dir == os.path.expanduser("~/test/path")
        finally:
            os.unlink(temp_path)

    def test_config_repr(self):
        """Test config string representation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            test_config = {}
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            repr_str = repr(config)
            assert "Config" in repr_str
            assert temp_path in repr_str
        finally:
            os.unlink(temp_path)
