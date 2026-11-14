# OmniTranscribe

<div align="center">

<img src="./omnitranscribe.png" alt="OmniTranscribe" width="400"/>

**A powerful multilingual audio transcription and translation tool**

English (Default) | [简体中文](./README.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-purple)](https://github.com/openai/whisper)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

OmniTranscribe is a comprehensive tool for processing audio and video content in **any language**. It combines state-of-the-art speech recognition with AI-powered translation to create a complete localization workflow for podcasts, lectures, meetings, videos, and more.

### What It Does

```
Input Audio/Video → Transcribe → Translate → Generate Output
   (Any Language)      (SRT)      (Any Language)  (SRT/VTT/LRC/MP4)
```

## Features

### Core Capabilities

- **Multilingual Transcription**: Convert audio/video to text in 99+ languages using OpenAI Whisper
- **Flexible Translation**: Translate subtitles between any languages using multiple AI models
- **Free Translation**: Built-in Google Translate support, no API key required
- **Format Conversion**: Support SRT, VTT, and LRC subtitle formats
- **Video Generation**: Create MP4 videos with embedded subtitles (default background included)
- **Smart Caching**: File hash-based caching to avoid re-transcription
- **Batch Processing**: Process multiple files automatically
- **Interactive Mode**: User-friendly command-line interface
- **Responsive GUI**: Web interface that adapts to screen size

### Supported Languages

Whisper supports the following 99 languages:

```
English, Chinese, Japanese, Korean, Spanish, French, German, Russian, Arabic, Hindi,
Portuguese, Italian, Dutch, Polish, Turkish, Vietnamese, Thai, Swedish, and more...
```

### Translation Services

- **Google Translate** ⭐ Free, no API key required (via deep-translator)
- **DeepSeek** (default) - Cost-effective and fast
- **Google Gemini** - High-quality translations
- **Alibaba Qwen** - Excellent for Chinese
- **Anthropic Claude** - Advanced AI reasoning
- **OpenAI GPT** - Industry-standard API
- **Custom APIs** - Support for OpenAI-compatible endpoints

### Supported Media Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Video**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM

## Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **FFmpeg** - Required for media processing

#### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/guaguastandup/OmniTranscribe.git
cd OmniTranscribe
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. Verify installation:
```bash
python run.py --help
```

## Quick Start

### GUI Mode (Recommended) ⭐

```bash
# Launch the graphical user interface
python run.py --gui

# Create a public link (accessible remotely)
python run.py --gui --share
```

The GUI interface provides:
- 📁 Drag & drop audio/video file upload
- 🌍 Visual language selection (99+ languages supported)
- ⚙️ Model and device settings
- 📤 Output format selection (SRT / VTT subtitles / MP4 video / MP3 audio)
- 🖼️ Background image and cover upload
- 🎵 Author information editing (title, artist, album)
- 📊 Real-time processing progress display

### Interactive Mode (Command Line)

```bash
python run.py
```

Follow the prompts to configure your settings and process files.

### Simple Usage (Command Line)

```bash
# Process audio file with smart defaults
python run.py audio_file.mp3

# Process video file
python run.py video_file.mp4

# Specify source language
python run.py audio_file.mp3 --language ja

# Specify both source and target language
python run.py audio_file.mp3 --language en --target-language zh
```

### Language Codes

Common language codes:
- `auto` - Auto-detection
- `en` - English
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `es` - Spanish
- `fr` - French
- `de` - German
- `ru` - Russian

## Usage Examples

### Multilingual Transcription

```bash
# Transcribe Japanese audio
python run.py japanese_audio.mp3 --language ja

# Transcribe Spanish video
python run.py spanish_video.mp4 --language es

# Auto-detect language
python run.py audio.mp3 --language auto

# Use specific Whisper model
python run.py audio.mp3 --model medium --language en
```

### Translation

```bash
# Translate to Chinese (default)
python run.py audio.mp3 --language en

# Translate to specific target language
python run.py audio.mp3 --language ja --target-language en

# Use different translation service
python run.py audio.mp3 --translation-model gemini
python run.py audio.mp3 --translation-model qwen
python run.py audio.mp3 --translation-model claude
python run.py audio.mp3 --translation-model gpt

# Custom API endpoint
python run.py audio.mp3 --translation-model custom \
  --translation-url "https://your-api.com/v1" \
  --translation-api-key "your-key"
```

### GPU Acceleration

```bash
# Use GPU acceleration
python run.py audio.mp3 --device cuda --language en

# Apple Silicon (M1/M2/M3) acceleration
python run.py audio.mp3 --device mps --language ja
```

### Subtitle Format Conversion

```bash
# Convert SRT to VTT or LRC
python run.py audio.mp3 --convert-to vtt
python run.py audio.mp3 --convert-to lrc

# Convert existing subtitle file only
python run.py --convert-only input.srt --convert-to vtt
```

### Proper-Noun Hints

```bash
# Improve recognition of names and jargon with a terminology hint
python run.py audio.mp3 --initial-prompt "OmniTranscribe, Whisper, GPT-4o"
```

### Video Generation

```bash
# Generate MP4 with subtitles
python run.py audio.mp3 --generate-video --background-image image.jpg

# Specify subtitle position
python run.py audio.mp3 --generate-video --background-image image.jpg --subtitle-position top
```

### Batch Processing

```bash
# Process all files in a directory
python run.py --batch /path/to/media/files --language auto

# Process recursively
python run.py --batch /path/to/files --recursive --language auto

# Delete original video files to save space
python run.py --batch /path/to/videos --delete-video-files
```

### Preset Modes

```bash
# Fast processing (tiny model)
python run.py --fast audio_file.mp3

# High quality (large model)
python run.py --quality audio_file.mp3

# GPU accelerated
python run.py --gpu audio_file.mp3
```

### Cache Management

```bash
# View cache statistics
python run.py --cache-stats

# Clear all cache
python run.py --clear-cache

# Disable cache (force re-transcription)
python run.py audio.mp3 --no-cache
```

### Transcription Only (No Translation)

```bash
# Transcribe only, skip translation
python run.py audio.mp3 --target-language none

# GUI mode: select "No Translation" option
python run.py --gui
```

## Configuration

### Environment Variables (.env)

```bash
# DeepSeek API (Recommended)
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat

# Google Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash

# Alibaba Qwen
QWEN_API_KEY=sk-your-qwen-key
QWEN_MODEL=qwen-plus

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-claude-key
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# OpenAI GPT
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini

# Default service
TRANSLATION_MODEL=deepseek

# Whisper defaults
WHISPER_MODEL=medium
WHISPER_LANGUAGE=auto
WHISPER_TARGET_LANGUAGE=zh
```

## Project Structure

```
OmniTranscribe/
├── run.py                     # Launcher script
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── README.md                  # Chinese documentation
├── README_EN.md               # English documentation
├── LICENSE                    # MIT License
├── CHANGELOG.md               # Changelog
├── CONTRIBUTING.md            # Contributing guide
├── prompt.md                  # Translation prompt
├── assets/                    # Default resources
│   ├── default_background.png  # Default video background
│   └── generate_default_bg.py  # Background generator
└── omnitranscribe/           # Source package (pip install .)
    ├── __init__.py
    ├── main.py                # Main entry point
    ├── transcribe.py          # Audio transcription
    ├── translator.py          # AI translation (with Google Translate)
    ├── converter.py           # Subtitle converter
    ├── video_converter.py     # Video to audio
    ├── final_video_generator.py   # Video generator
    ├── simple_mp3_embedder.py     # MP3 metadata
    ├── batch_processor.py     # Batch processing
    ├── cache.py               # Cache management
    ├── interactive.py         # Interactive CLI
    ├── config.py              # Configuration
    └── gui.py                 # Web GUI interface
```

## Troubleshooting

### FFmpeg not found
```
Error: ffmpeg is not installed or not in PATH
```
**Solution**: Install FFmpeg (see [Installation](#installation))

### API key errors
```
Error: DEEPSEEK_API_KEY not found in environment variables
```
**Solution**: Copy `.env.example` to `.env` and add your API keys

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use a smaller Whisper model or switch to CPU: `--device cpu`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for multilingual speech recognition
- [FFmpeg](https://ffmpeg.org/) for media processing
- All translation service providers (DeepSeek, Google, Alibaba, Anthropic, OpenAI)
- [ChillType fonts](https://github.com/Warren2060) (ChillDuanSans / ChillHuoFangSong, [OFL 1.1](./FONTS.md)) for the bundled subtitle fonts

---

<div align="center">

Made with ❤️ for multilingual audio content lovers

[![GitHub stars](https://img.shields.io/github/stars/guaguastandup/OmniTranscribe?style=social)](https://github.com/guaguastandup/OmniTranscribe/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/guaguastandup/OmniTranscribe?style=social)](https://github.com/guaguastandup/OmniTranscribe/network/members)

</div>
