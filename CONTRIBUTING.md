# Contributing to OmniTranscribe

Thank you for your interest in contributing to OmniTranscribe! This document provides guidelines and instructions for contributing to the project.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**: Be specific about what the problem is
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**:
  - OS and version
  - Python version
  - Relevant dependency versions
- **Screenshots/logs**: If applicable, include error messages or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title and description**: What the enhancement is
- **Use case**: Why this enhancement would be useful
- **Proposed solution**: How you envision the implementation (if you have ideas)

### Pull Requests

1. Fork the repository and create your branch from `main`
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make your changes with clear, descriptive commit messages
4. Write/update tests if applicable
5. Ensure your code follows the existing style
6. Update documentation as needed
7. Push to your fork and submit a pull request

#### Pull Request Guidelines

- **One feature per PR**: Keep changes focused
- **Clear description**: Explain what you changed and why
- **Reference issues**: Include "Fixes #123" if it resolves an issue
- **Tests**: Include tests for new functionality
- **Documentation**: Update README and docstrings as needed

## Development Setup

### Cloning the Repository

```bash
git clone https://github.com/guaguastandup/OmniTranscribe.git
cd OmniTranscribe
```

### Creating a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests (when available)
python -m pytest

# Run specific test file
python -m pytest tests/test_module.py
```

## Coding Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and reasonably sized
- Add comments for complex logic

### Example

```python
def transcribe_audio(file_path: str, model: str = "base") -> dict:
    """
    Transcribe audio file using Whisper.

    Args:
        file_path: Path to the audio file
        model: Whisper model size (tiny/base/small/medium/large)

    Returns:
        Dictionary containing transcription results

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If model size is invalid
    """
    # Implementation here
    pass
```

## Project Structure

```
OmniTranscribe/
├── main.py                    # Main entry point
├── transcribe.py              # Audio transcription module
├── translator.py              # Translation module
├── converter.py               # Format conversion module
├── video_converter.py         # Video processing
├── final_video_generator.py   # Video generation
├── batch_processor.py         # Batch processing
├── interactive.py             # Interactive interface
├── mp3_creator.py             # MP3 utilities
├── simple_mp3_embedder.py     # MP3 metadata
├── config.py                  # Configuration
├── clean_existing_files.py    # Cleanup utilities
├── manual_clean.py            # Manual cleanup
├── quick_start.py             # Quick start wizard
├── prompt.md                  # Translation prompts
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
└── run.sh                     # Batch script
```

## Adding New Features

When adding new features:

1. **Discuss first**: Open an issue to discuss the feature before implementing
2. **Modular design**: Keep new code in separate, focused modules
3. **Error handling**: Include proper error handling and user-friendly messages
4. **Documentation**: Update README.md and add docstrings
5. **Backwards compatibility**: Maintain compatibility with existing functionality

## Translation Service Support

To add support for a new translation service:

1. Add API configuration to `.env.example`
2. Implement the service in `translator.py`
3. Update the `--translation-model` argument in `main.py`
4. Add documentation to README.md
5. Test the implementation

Example:

```python
def translate_with_new_service(text: str, api_key: str) -> str:
    """Translate text using new service."""
    # Implementation
    pass
```

## Code Review Process

1. All submissions require review before merging
2. Address review comments promptly
3. Keep discussions constructive and focused
4. Request re-review after making changes

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Getting Help

- Check existing [documentation](README.md)
- Search [existing issues](https://github.com/guaguastandup/OmniTranscribe/issues)
- Ask questions in [discussions](https://github.com/guaguastandup/OmniTranscribe/discussions)

## License

By contributing to OmniTranscribe, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to OmniTranscribe! Your contributions help make this project better for everyone.
