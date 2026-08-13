# OmniTranscribe desktop packaging

The desktop build keeps the existing Python/Whisper core, freezes it as a one-folder application with PyInstaller, bundles `ffmpeg`/`ffprobe`, and opens the Gradio workspace inside a native `pywebview` window.

## Local prerequisites

- Python 3.11 recommended for release builds.
- FFmpeg and ffprobe available on `PATH`.
- Install desktop build dependencies with `pip install -r requirements-desktop.txt`.

## Local build

Create a `bin/` directory in the repository root and copy the platform-native `ffmpeg` and `ffprobe` executables into it. Then run:

```bash
pyinstaller --clean --noconfirm packaging/OmniTranscribe.spec
```

Outputs:

- Windows: `dist/OmniTranscribe/`
- macOS: `dist/OmniTranscribe.app`

The GitHub Actions workflow `.github/workflows/desktop-release.yml` performs platform-specific FFmpeg collection automatically and creates an Inno Setup `.exe` on Windows and a `.dmg` on macOS.

## Release

After the workflow is merged to the default branch, push a version tag such as `v0.1.0`. The workflow builds Windows and macOS artifacts and attaches them to the corresponding GitHub Release. The current workflow produces unsigned builds; code signing/notarization should be added before treating the macOS package as a polished public release.
