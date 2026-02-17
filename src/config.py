#!/usr/bin/env python3
"""
Configuration management for TranscribeX audio transcription tool.

This module handles loading, saving, and managing user preferences
to reduce the need for repetitive command line arguments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for transcription settings."""

    DEFAULT_CONFIG = {
        "transcription": {
            "model": "base",
            "device": "auto",
            "language": "auto",
            "strict_mode": False
        },
        "translation": {
            "model": "deepseek-chat",
            "target_language": "zh",
            "chunk_size": 20,
            "prompt_file": "prompt.md"
        },
        "output": {
            "default_dir": "output",
            "format": "srt",
            "keep_video_files": True
        },
        "batch": {
            "recursive": True,
            "max_workers": 1,
            "skip_existing": False
        },
        "ui": {
            "interactive_mode": True,
            "show_progress": True,
            "auto_open_output": False
        }
    }

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to config file (default: ~/.config/transcribex/config.json)
        """
        if config_file is None:
            config_dir = Path.home() / ".config" / "omnitranscribe"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, creating default if doesn't exist."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # Merge with defaults to ensure all keys exist
                config = self._deep_merge(self.DEFAULT_CONFIG, loaded_config)
                return config
            except Exception as e:
                print(f"Warning: Failed to load config file, using defaults: {e}")

        return self.DEFAULT_CONFIG.copy()

    def _deep_merge(self, default: Dict, loaded: Dict) -> Dict:
        """Deep merge two dictionaries, preserving defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def save(self):
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Failed to save config file: {e}")

    def get(self, key_path: str, default=None):
        """
        Get configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path like 'transcription.model'
            default: Default value if key not found
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value):
        """
        Set configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path like 'transcription.model'
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration."""
        return self.config.copy()

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()

    def print_config(self):
        """Print current configuration in a readable format."""
        print("Current Configuration:")
        print("=" * 40)

        def print_section(section: Dict, indent: int = 0):
            for key, value in section.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_section(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        print_section(self.config)
        print("=" * 40)
        print(f"Config file: {self.config_file}")

# Global config instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def save_config():
    """Save global configuration."""
    global _config
    if _config is not None:
        _config.save()