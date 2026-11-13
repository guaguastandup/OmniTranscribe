# Changelog

All notable changes to OmniTranscribe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-11-13

### Added
- Resumable product workflow: SQLite project store, chunked transcription and
  translation with checkpoints, in-app subtitle proofreading table
  (`src/resumable_pipeline.py`, `src/project_store.py`, `src/product_gui.py`)
- Desktop packaging: PyInstaller spec, Windows installer and macOS DMG
  release workflow, native window via pywebview (`run.py --desktop`)
- API key storage with OS keyring support and a permission-restricted file
  fallback (`src/app_settings.py`)
- ruff lint configuration and CI enforcement; unit tests for the converter,
  cache, config and translator helpers (32 tests total)

### Fixed
- `--language auto` no longer crashes Whisper ("auto" now maps to None for
  auto-detection); batch mode no longer silently transcribes everything as
  Japanese or translates to Chinese regardless of the requested languages
- Translation providers: Claude and Gemini now use their documented
  OpenAI-compatible endpoints; `--fast/--quality/--gpu` presets, and
  `--subtitle-fontsize/--subtitle-position`, and the `WHISPER_*` /
  `TRANSLATION_MODEL` `.env` variables now actually take effect
- `--target-language none` works end to end (format conversion, video and
  MP3 exports fall back to the original subtitles) and no longer requires
  an API key
- Transcription no longer crashes on Apple Silicon (MPS): whisper's word
  alignment uses float64, which MPS does not support; `word_timestamps` is
  now opt-in (SRT output only needs segment timestamps)
- Subtitle burning: the ffmpeg filter now uses the explicit `filename=`
  syntax (ffmpeg 8+ rejects the old quoted positional form) and fails with
  a clear message when the installed ffmpeg lacks libass
- Replaced `eval()` on ffprobe output and deprecated `tempfile.mktemp()`;
  bare `except:` clauses removed; translation requests get a timeout
  (`TRANSLATION_TIMEOUT`, default 120s)

### Removed
- The legacy one-shot Gradio GUI (`--legacy-gui`); the project-based GUI
  (`run.py --gui`) covers all of its functionality with resumable
  transcription, subtitle proofreading and more export options

### Changed
- **Pipeline convergence**: the CLI single-file mode now delegates to the
  same orchestration as batch mode (`BatchProcessor`), which also gained
  MP3-metadata embedding, per-request translation overrides and a lazy
  translator (no API key needed when translation is skipped); unknown
  translation service names are validated up front with a clear error
- **Packaging**: `src/` renamed to the installable `omnitranscribe/` package;
  added `pyproject.toml` with `omnitranscribe` / `omnitranscribe-gui` console
  scripts (`pip install .` now works); internal imports are package-relative
- Product GUI visual refresh: gradient hero, semantic stage pills, progress
  bars, themed buttons and tabs
- Whisper models are cached per (model, device) so resuming a project no
  longer reloads the model; transcription cache hashes each file once
- Removed ~240 lines of dead Japanese-strict fallback code from
  `src/transcribe.py`; README banner compressed from 5.5MB to 67KB

### Compliance
- Bundled ChillType fonts documented and shipped with their SIL OFL 1.1
  license text (`FONTS.md`, `OFL-1.1.txt`)

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
