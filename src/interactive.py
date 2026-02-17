#!/usr/bin/env python3
"""
Interactive mode for audio transcription tool.

This module provides an interactive command-line interface to guide users
through the transcription process without needing to remember complex commands.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import inquirer, but provide fallback if not available
try:
    import inquirer
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False

from config import get_config, save_config

class InteractiveMode:
    """Interactive mode for user-friendly operation."""

    def __init__(self):
        self.config = get_config()

    def run(self):
        """Run the interactive mode."""
        print("🎵 Audio Transcription Tool - Interactive Mode")
        print("=" * 50)
        print("Let's configure your transcription settings step by step.")
        print()

        # Main mode selection
        mode = self._select_mode()

        if mode == "single":
            return self._single_file_mode()
        elif mode == "batch":
            return self._batch_mode()
        elif mode == "convert":
            return self._convert_mode()
        elif mode == "translate":
            return self._translate_mode()
        elif mode == "config":
            return self._config_mode()
        elif mode == "presets":
            return self._presets_mode()

    def _select_mode(self) -> str:
        """Select the operation mode."""
        questions = [
            inquirer.List(
                'mode',
                message="What would you like to do?",
                choices=[
                    ('🎵 Transcribe single audio/video file', 'single'),
                    ('📁 Process multiple files (batch)', 'batch'),
                    ('🔄 Convert subtitle format', 'convert'),
                    ('🌐 Translate existing subtitles', 'translate'),
                    ('⚙️  Configure settings', 'config'),
                    ('⚡ Use preset configurations', 'presets')
                ],
                carousel=True
            )
        ]
        answers = inquirer.prompt(questions)
        return answers['mode'] if answers else 'single'

    def _single_file_mode(self) -> dict:
        """Configure single file transcription."""
        print("\n🎵 Single File Transcription")
        print("-" * 30)

        # File selection
        file_path = self._select_file("Select audio/video file to transcribe")

        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        is_video = Path(file_path).suffix.lower() in video_extensions

        # Output configuration
        output_config = self._configure_output()

        # Transcription settings
        trans_config = self._configure_transcription()

        # Translation settings
        translate_config = self._configure_translation()

        return {
            'mode': 'single',
            'file_path': file_path,
            'is_video': is_video,
            'output': output_config,
            'transcription': trans_config,
            'translation': translate_config
        }

    def _batch_mode(self) -> dict:
        """Configure batch processing."""
        print("\n📁 Batch Processing")
        print("-" * 30)

        # Directory selection
        input_path = self._select_directory("Select directory with media files")

        # Output configuration
        output_config = self._configure_output()

        # Transcription settings
        trans_config = self._configure_transcription()

        # Translation settings
        translate_config = self._configure_translation()

        # Batch-specific settings
        batch_config = self._configure_batch_settings()

        return {
            'mode': 'batch',
            'input_path': input_path,
            'output': output_config,
            'transcription': trans_config,
            'translation': translate_config,
            'batch': batch_config
        }

    def _convert_mode(self) -> dict:
        """Configure subtitle conversion."""
        print("\n🔄 Subtitle Conversion")
        print("-" * 30)

        file_path = self._select_file("Select subtitle file to convert", subtitle_only=True)

        questions = [
            inquirer.List(
                'format',
                message="Convert to format:",
                choices=[('SRT', 'srt'), ('VTT', 'vtt'), ('LRC', 'lrc')],
                default=self.config.get('output.format', 'srt')
            )
        ]
        answers = inquirer.prompt(questions)

        return {
            'mode': 'convert',
            'file_path': file_path,
            'format': answers['format']
        }

    def _translate_mode(self) -> dict:
        """Configure translation of existing subtitles."""
        print("\n🌐 Translate Existing Subtitles")
        print("-" * 30)

        file_path = self._select_file("Select Japanese subtitle file to translate", subtitle_only=True)

        translate_config = self._configure_translation()

        return {
            'mode': 'translate',
            'file_path': file_path,
            'translation': translate_config
        }

    def _config_mode(self) -> dict:
        """Configure default settings."""
        print("\n⚙️  Configuration Settings")
        print("-" * 30)

        print("Current configuration:")
        self.config.print_config()

        questions = [
            inquirer.List(
                'action',
                message="What would you like to configure?",
                choices=[
                    ('🎵 Transcription settings', 'transcription'),
                    ('🌐 Translation settings', 'translation'),
                    ('📁 Output settings', 'output'),
                    ('📦 Batch settings', 'batch'),
                    ('💾 Save current settings as defaults', 'save'),
                    ('🔄 Reset to defaults', 'reset'),
                    ('↩️  Back to main menu', 'back')
                ],
                carousel=True
            )
        ]
        answers = inquirer.prompt(questions)

        action = answers['action']

        if action == 'transcription':
            self._configure_transcription(save_to_config=True)
            return self._config_mode()
        elif action == 'translation':
            self._configure_translation(save_to_config=True)
            return self._config_mode()
        elif action == 'output':
            self._configure_output(save_to_config=True)
            return self._config_mode()
        elif action == 'batch':
            self._configure_batch_settings(save_to_config=True)
            return self._config_mode()
        elif action == 'save':
            self.config.save()
            print("✅ Settings saved successfully!")
            return {'mode': 'config_saved'}
        elif action == 'reset':
            self.config.reset_to_defaults()
            print("✅ Settings reset to defaults!")
            return {'mode': 'config_reset'}
        else:
            return self.run()

    def _presets_mode(self) -> dict:
        """Select preset configurations."""
        print("\n⚡ Preset Configurations")
        print("-" * 30)

        presets = {
            'fast': {
                'name': '🚀 Fast Processing',
                'description': 'Quick transcription with smaller model',
                'config': {
                    'transcription': {'model': 'tiny', 'device': 'auto'},
                    'translation': {'chunk_size': 30},
                    'batch': {'max_workers': 1}
                }
            },
            'balanced': {
                'name': '⚖️  Balanced',
                'description': 'Good balance of speed and accuracy',
                'config': {
                    'transcription': {'model': 'base', 'device': 'auto'},
                    'translation': {'chunk_size': 20},
                    'batch': {'max_workers': 1}
                }
            },
            'quality': {
                'name': '🎯 High Quality',
                'description': 'Best accuracy with larger model',
                'config': {
                    'transcription': {'model': 'large', 'device': 'auto'},
                    'translation': {'chunk_size': 15},
                    'batch': {'max_workers': 1}
                }
            },
            'gpu': {
                'name': '🔥 GPU Accelerated',
                'description': 'Maximum speed with GPU',
                'config': {
                    'transcription': {'model': 'medium', 'device': 'auto'},
                    'translation': {'chunk_size': 25},
                    'batch': {'max_workers': 1}
                }
            }
        }

        choices = []
        for key, preset in presets.items():
            choices.append((f"{preset['name']} - {preset['description']}", key))

        choices.append(('↩️  Back to main menu', 'back'))

        questions = [
            inquirer.List(
                'preset',
                message="Select a preset:",
                choices=choices,
                carousel=True
            )
        ]
        answers = inquirer.prompt(questions)

        preset_key = answers['preset']
        if preset_key == 'back':
            return self.run()

        # Apply preset
        preset = presets[preset_key]
        for section, settings in preset['config'].items():
            for key, value in settings.items():
                self.config.set(f"{section}.{key}", value)

        print(f"✅ Applied preset: {preset['name']}")
        self.config.save()

        # Ask what to do next
        questions = [
            inquirer.List(
                'next',
                message="What would you like to do now?",
                choices=[
                    ('🎵 Start transcription with these settings', 'transcribe'),
                    ('⚙️  Fine-tune settings', 'config'),
                    ('↩️  Back to main menu', 'back')
                ],
                carousel=True
            )
        ]
        answers = inquirer.prompt(questions)

        if answers['next'] == 'transcribe':
            return self._select_mode()
        elif answers['next'] == 'config':
            return self._config_mode()
        else:
            return self.run()

    def _select_file(self, message: str, subtitle_only: bool = False) -> str:
        """Select a file interactively."""
        while True:
            file_path = input(f"{message}: ").strip().strip('"\'')

            if not file_path:
                print("❌ Please enter a file path.")
                continue

            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            if subtitle_only:
                subtitle_extensions = {'.srt', '.vtt', '.lrc'}
                if Path(file_path).suffix.lower() not in subtitle_extensions:
                    print(f"❌ Please select a subtitle file (SRT, VTT, or LRC)")
                    continue

            return file_path

    def _select_directory(self, message: str) -> str:
        """Select a directory interactively."""
        while True:
            dir_path = input(f"{message}: ").strip().strip('"\'')

            if not dir_path:
                print("❌ Please enter a directory path.")
                continue

            if not os.path.exists(dir_path):
                print(f"❌ Directory not found: {dir_path}")
                continue

            if not os.path.isdir(dir_path):
                print(f"❌ Not a directory: {dir_path}")
                continue

            return dir_path

    def _configure_transcription(self, save_to_config: bool = False) -> dict:
        """Configure transcription settings."""
        questions = [
            inquirer.List(
                'model',
                message="Whisper model size:",
                choices=[
                    ('Tiny (fastest, less accurate)', 'tiny'),
                    ('Base (balanced)', 'base'),
                    ('Small (good accuracy)', 'small'),
                    ('Medium (very accurate)', 'medium'),
                    ('Large (best accuracy, slow)', 'large')
                ],
                default=self.config.get('transcription.model', 'base')
            ),
            inquirer.List(
                'device',
                message="Processing device:",
                choices=[
                    ('Auto-detect (recommended)', 'auto'),
                    ('CPU', 'cpu'),
                    ('NVIDIA GPU (CUDA)', 'cuda'),
                    ('Apple Silicon GPU (MPS)', 'mps')
                ],
                default=self.config.get('transcription.device', 'auto')
            ),
            inquirer.Confirm(
                'force_japanese',
                message="Force Japanese transcription (prevents English output)?",
                default=self.config.get('transcription.force_japanese', True)
            )
        ]

        answers = inquirer.prompt(questions)

        if save_to_config:
            for key, value in answers.items():
                self.config.set(f'transcription.{key}', value)

        return answers

    def _configure_translation(self, save_to_config: bool = False) -> dict:
        """Configure translation settings."""
        questions = [
            inquirer.List(
                'model',
                message="Translation model:",
                choices=[
                    ('DeepSeek (recommended)', 'deepseek-chat'),
                    ('Gemini', 'gemini'),
                    ('Qwen', 'qwen'),
                    ('Claude', 'claude'),
                    ('GPT', 'gpt'),
                    ('Custom API', 'custom')
                ],
                default=self.config.get('translation.model', 'deepseek-chat')
            ),
            inquirer.Confirm(
                'enable_translation',
                message="Enable translation to Chinese?",
                default=True
            )
        ]

        answers = inquirer.prompt(questions)

        if not answers['enable_translation']:
            return {'enable_translation': False}

        # Ask for chunk size if translation is enabled
        chunk_questions = [
            inquirer.Text(
                'chunk_size',
                message="Translation chunk size (number of subtitles per request):",
                default=str(self.config.get('translation.chunk_size', 20)),
                validate=lambda _, x: x.isdigit() and int(x) > 0
            )
        ]
        chunk_answers = inquirer.prompt(chunk_questions)
        answers['chunk_size'] = int(chunk_answers['chunk_size'])

        if save_to_config:
            for key, value in answers.items():
                self.config.set(f'translation.{key}', value)

        return answers

    def _configure_output(self, save_to_config: bool = False) -> dict:
        """Configure output settings."""
        questions = [
            inquirer.Text(
                'output_dir',
                message="Output directory:",
                default=self.config.get('output.default_dir', 'output')
            ),
            inquirer.List(
                'format',
                message="Subtitle format:",
                choices=[('SRT', 'srt'), ('VTT', 'vtt'), ('LRC', 'lrc')],
                default=self.config.get('output.format', 'srt')
            )
        ]

        answers = inquirer.prompt(questions)

        if save_to_config:
            for key, value in answers.items():
                self.config.set(f'output.{key}', value)

        return answers

    def _configure_batch_settings(self, save_to_config: bool = False) -> dict:
        """Configure batch-specific settings."""
        questions = [
            inquirer.Confirm(
                'recursive',
                message="Search subdirectories recursively?",
                default=self.config.get('batch.recursive', True)
            ),
            inquirer.Confirm(
                'skip_existing',
                message="Skip files that already have processed output?",
                default=self.config.get('batch.skip_existing', False)
            ),
            inquirer.Confirm(
                'keep_video_files',
                message="Keep original video files after conversion?",
                default=self.config.get('output.keep_video_files', True)
            )
        ]

        answers = inquirer.prompt(questions)

        if save_to_config:
            for key, value in answers.items():
                self.config.set(f'batch.{key}', value)

        return answers

def run_interactive():
    """Run the interactive mode."""
    if not INQUIRER_AVAILABLE:
        print("❌ Interactive mode requires 'inquirer' package.")
        print("Please install it with: pip install inquirer")
        print()
        print("Alternatively, use these simple commands:")
        print("  python main.py --fast audio_file.mp3      # Fast processing")
        print("  python main.py --quality audio_file.mp3   # High quality")
        print("  python main.py --gpu audio_file.mp3       # GPU accelerated")
        print("  python main.py audio_file.mp3             # Use defaults")
        sys.exit(1)

    mode = InteractiveMode()
    return mode.run()