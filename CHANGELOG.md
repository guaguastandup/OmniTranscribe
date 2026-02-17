# Changelog

All notable changes to OmniTranscribe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-02-17

### Major Changes
- **Project renamed** to OmniTranscribe
- **Multilingual support**: Now supports 99+ languages instead of just Japanese
- Added auto-detection for source language
- Added flexible target language selection for translation

### Changed
- Removed redundant video generator modules (video_generator.py, basic_video_generator.py, simple_video_generator.py, minimal_video_generator.py)
- Removed redundant MP3 metadata embedder (mp3_metadata_embedder.py)
- Consolidated to use `final_video_generator.py` as the single video generation solution
- Consolidated to use `simple_mp3_embedder.py` as the single MP3 metadata solution
- Updated requirements.txt to remove unused dependencies (moviepy, eyed3, mutagen)
- `--force-japanese` replaced with `--strict-mode` for language-agnostic strict transcription
- Default language changed from `ja` to `auto` for automatic detection
- Added `--target-language` option for flexible translation targets
- Added `--language` (-l) shorthand for source language
- Added `--target-language` (-t) shorthand for target language

### Added
- Professional README.md with comprehensive multilingual documentation
- MIT License
- Contributing guidelines (CONTRIBUTING.md)
- Changelog (CHANGELOG.md)
- Support for 99+ languages through OpenAI Whisper

### Code Cleanup
- Removed approximately 900 lines of redundant code
- Cleaned up unused dependencies
- Improved project structure for better maintainability

## [1.0.0] - Initial Release

### Features
- Audio transcription using OpenAI Whisper
- AI-powered translation with multiple service support
- Support for multiple translation services (DeepSeek, Gemini, Qwen, Claude, GPT, Custom APIs)
- Subtitle format conversion (SRT, VTT, LRC)
- Video to audio conversion
- MP4 video generation with embedded subtitles
- MP3 metadata embedding (lyrics and cover art)
- Batch processing with recursive directory support
- Interactive mode for beginners
- Configuration persistence
- Preset modes (--fast, --quality, --gpu)
- GPU acceleration support (CUDA, MPS for Apple Silicon)

### Documentation
- Quick start guide
- MP3 creation guide
- MP3 metadata guide
- Translation models documentation

---

## Version Format

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes
