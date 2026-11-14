#!/usr/bin/env python3
"""
Final Video Generator with Subtitles

This module creates MP4 videos with synchronized subtitles from:
- MP3 audio file
- SRT subtitle file
- Background image file
- Subtitle display at bottom of image

Uses direct FFmpeg commands with proper subtitle rendering.
"""

import os
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Optional
from .converter import SubtitleConverter


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


class FinalVideoGenerator:
    """Generate MP4 videos with synchronized subtitles using FFmpeg."""

    def __init__(self, custom_font_path: Optional[str] = None):
        """
        Initialize the video generator.

        Args:
            custom_font_path: Optional path to a custom font file (TTF, OTF, TTC)
        """
        self.font_path = self._get_chinese_font(custom_font_path)

    def _get_chinese_font(self, custom_font_path: Optional[str] = None) -> str:
        """
        Get the font path for Chinese characters on the current platform.

        Args:
            custom_font_path: Optional path to a custom font file

        Returns:
            Font file path that supports Chinese characters (for FFmpeg subtitles filter)
        """
        # First check if user provided a custom font
        if custom_font_path:
            if os.path.exists(custom_font_path):
                print(f"Using custom font: {custom_font_path}")
                return custom_font_path
            else:
                print(f"Warning: Custom font not found at {custom_font_path}, falling back to default font")

        # Check environment variable for custom font
        env_font = os.environ.get('OMNITRANSCRIBE_FONT_PATH')
        if env_font and os.path.exists(env_font):
            print(f"Using font from environment variable: {env_font}")
            return env_font

        # Check project default fonts
        project_root = _get_project_root()
        default_fonts = [
            project_root / "ChillDuanSansVF.ttf",
            project_root / "ChillHuoFangSong_Regular.otf",
        ]
        for font_path in default_fonts:
            if font_path.exists():
                print(f"Using project default font: {font_path}")
                return str(font_path)

        system = platform.system()

        if system == "Darwin":  # macOS
            # Use actual font file paths for macOS (more reliable than font names)
            font_paths = [
                # PingFang SC (Standard Chinese font)
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Supplemental/PingFang.ttc",
                "/Library/Fonts/PingFang.ttc",
                # Hiragino Sans GB
                "/System/Library/Fonts/hiroshge.ttc",
                "/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc",
                # STHeiti
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                # Hei (SC)
                "/System/Library/Fonts/STHeitiSC-Light.ttc",
                "/System/Library/Fonts/STHeitiSC-Medium.ttc",
                # Arial Unicode MS
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/Arial Unicode.ttf",
            ]

            # Find first existing font
            for font_path in font_paths:
                if os.path.exists(font_path):
                    print(f"Using system font: {font_path}")
                    return font_path

            # Fallback: Use font name if no file found
            print("Warning: No font file found, trying font name 'PingFang SC'")
            return "PingFang SC"

        elif system == "Linux":
            # Common Linux font paths
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]

            # Find first existing font
            for font_path in font_paths:
                if os.path.exists(font_path):
                    print(f"Using system font: {font_path}")
                    return font_path

            return "WenQuanYi Zen Hei"

        elif system == "Windows":
            # Windows font paths (with user profile support)
            fonts_dir = os.path.join(os.environ.get('SYSTEMROOT', r'C:\Windows'), 'Fonts')

            font_paths = [
                os.path.join(fonts_dir, 'msyh.ttc'),  # Microsoft YaHei
                os.path.join(fonts_dir, 'msyhbd.ttc'),  # Microsoft YaHei Bold
                os.path.join(fonts_dir, 'simsun.ttc'),  # SimSun
                os.path.join(fonts_dir, 'simhei.ttf'),  # SimHei
                os.path.join(fonts_dir, 'dengxian.ttf'),  # DengXian
                os.path.join(fonts_dir, 'simkai.ttf'),  # KaiTi
            ]

            # Find first existing font
            for font_path in font_paths:
                if os.path.exists(font_path):
                    print(f"Using system font: {font_path}")
                    return font_path

            return "Microsoft YaHei"

        else:
            # Fallback for other systems
            print("Warning: Unknown platform, using default font (may not display Chinese correctly)")
            return ""

    @staticmethod
    def _subtitles_filter_available() -> bool:
        """Check whether this ffmpeg build ships the libass subtitles filter."""
        try:
            probe = subprocess.run(
                ['ffmpeg', '-hide_banner', '-h', 'filter=subtitles'],
                capture_output=True, text=True
            )
            combined = probe.stdout + probe.stderr
            # ffmpeg exits 0 even for unknown filters; the verdict is in the text
            return 'Unknown filter' not in combined and 'subtitles' in combined.lower()
        except (OSError, subprocess.SubprocessError):
            return False

    def create_video_with_subtitles(self,
                                   audio_path: str,
                                   srt_path: str,
                                   image_path: str,
                                   output_path: str,
                                   max_duration: Optional[float] = None,
                                   font_size: int = 28,
                                   subtitle_position: str = "bottom") -> str:
        """
        Create MP4 video with synchronized subtitles.

        Args:
            audio_path: Path to MP3 audio file
            srt_path: Path to SRT subtitle file
            image_path: Path to background image file
            output_path: Path for output MP4 file
            max_duration: Maximum duration (None for full audio)
            font_size: Subtitle font size (libass FontSize)
            subtitle_position: Subtitle position (top, bottom, center)

        Returns:
            Path to generated MP4 file
        """
        # libass numpad alignment: 8=top-center, 5=middle-center, 2=bottom-center
        alignment = {"top": 8, "center": 5, "bottom": 2}.get(subtitle_position, 2)
        # Validate input files
        for file_path, file_type in [(audio_path, "Audio"), (srt_path, "Subtitle"), (image_path, "Image")]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_type} file not found: {file_path}")

        print("Processing files for video generation...")

        # Get audio duration using ffmpeg
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', audio_path],
                capture_output=True, text=True, check=True
            )
            full_duration = float(result.stdout.strip())
            video_duration = full_duration if max_duration is None else min(full_duration, max_duration)
            print(f"Audio duration: {full_duration:.2f} seconds (using {video_duration:.2f}s)")
        except Exception as e:
            raise RuntimeError(f"Failed to get audio duration: {str(e)}")

        # Create temporary files
        temp_audio_path = None
        if max_duration is not None and video_duration < full_duration:
            # Create shorter audio clip (NamedTemporaryFile, not the deprecated mktemp)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
            print(f"Creating {video_duration}s audio clip...")
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-t', str(video_duration),
                '-acodec', 'copy',
                temp_audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            audio_to_use = temp_audio_path
        else:
            audio_to_use = audio_path

        try:
            print("Creating video with subtitles...")

            srt_abs_path = os.path.abspath(srt_path)

            # Verify the SRT file exists before passing to FFmpeg
            if not os.path.exists(srt_abs_path):
                raise FileNotFoundError(f"SRT file not found: {srt_abs_path}")

            # The subtitles filter requires libass; give a clear error instead
            # of an opaque ffmpeg exit code when the build lacks it.
            if not self._subtitles_filter_available():
                raise RuntimeError(
                    "当前 FFmpeg 没有 subtitles 字幕滤镜（需要 libass 支持）。"
                    "请安装完整版 FFmpeg（如 macOS: brew install ffmpeg）后重试。"
                )

            print(f"Using font: {self.font_path}")

            # Prepare environment for FFmpeg
            env = os.environ.copy()

            # Explicit "filename=" key keeps the filter chain parseable on
            # ffmpeg 8+ (a quoted positional followed by :key=value no longer
            # parses there). Colons and quotes in the path must be escaped
            # for the filter parser (Windows drive letters use colons).
            filter_path = (
                srt_abs_path.replace("\\", "/")
                .replace(":", r"\:")
                .replace("'", r"\'")
            )

            if self.font_path and os.path.exists(self.font_path):
                font_dir = os.path.dirname(self.font_path)
                font_file = os.path.basename(self.font_path)

                # Get font name without extension (fontconfig uses this)
                font_name = os.path.splitext(font_file)[0]
                print(f"Font name for FFmpeg: {font_name}")
                print(f"Font directory: {font_dir}")

                # Set environment variables for libass/fontconfig
                # These tell FFmpeg where to look for fonts
                env['ASS_FONTPATH'] = font_dir

                subtitle_style = (
                    f"FontName={font_name},Fontsize={font_size},"
                    f"PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Alignment={alignment}"
                )
            else:
                subtitle_style = f"Fontsize={font_size},Alignment={alignment}"

            # Single encoding pass: scale/pad the looping image and burn the
            # subtitles in one filter chain. This replaces the previous
            # two-pass approach (base video, then subtitle re-encode), which
            # encoded the video twice.
            video_filter = (
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
                f"subtitles=filename='{filter_path}':force_style='{subtitle_style}'"
            )
            print(f"Video filter: {video_filter}")

            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-i', audio_to_use,
                '-t', str(video_duration),
                '-vf', video_filter,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-r', '24',
                '-shortest',
                output_path
            ]

            subprocess.run(cmd, check=True, env=env)
            print(f"Video with subtitles completed: {output_path}")
            return output_path

        finally:
            # Clean up temporary files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass

    def create_video_from_existing_files(self,
                                       audio_path: str,
                                       lyrics_path: str,
                                       image_path: str,
                                       output_path: str,
                                       **kwargs) -> str:
        """
        Create video from existing audio, lyrics (LRC or SRT), and image.

        Args:
            audio_path: Path to audio file
            lyrics_path: Path to lyrics file (LRC or SRT format)
            image_path: Path to background image
            output_path: Path for output video
            **kwargs: Additional parameters

        Returns:
            Path to generated MP4 file
        """
        # Determine lyrics format and convert if needed
        lyrics_ext = Path(lyrics_path).suffix.lower()

        temp_srt_fd = None
        temp_srt_path = None

        if lyrics_ext == '.srt':
            srt_path = lyrics_path
        elif lyrics_ext == '.lrc':
            # Convert LRC to SRT using a proper temporary file
            converter = SubtitleConverter()
            # Use mkstemp instead of mktemp for safer temporary file creation
            temp_srt_fd, temp_srt_path = tempfile.mkstemp(suffix='.srt', text=True)
            try:
                with open(lyrics_path, 'r', encoding='utf-8') as f:
                    lrc_content = f.read()
                srt_content = converter.lrc_to_srt(lrc_content)
                # Write using the file descriptor
                os.write(temp_srt_fd, srt_content.encode('utf-8'))
                os.close(temp_srt_fd)
                temp_srt_fd = None
            except Exception:
                # Clean up on error
                if temp_srt_fd is not None:
                    os.close(temp_srt_fd)
                if temp_srt_path and os.path.exists(temp_srt_path):
                    os.unlink(temp_srt_path)
                raise
            srt_path = temp_srt_path
        else:
            raise ValueError(f"Unsupported lyrics format: {lyrics_ext}")

        try:
            return self.create_video_with_subtitles(audio_path, srt_path, image_path, output_path, **kwargs)
        finally:
            # Clean up temporary SRT file if created
            if temp_srt_path and os.path.exists(temp_srt_path):
                try:
                    os.unlink(temp_srt_path)
                except OSError:
                    pass


def generate_video_with_subtitles(audio_path: str,
                                 subtitles_path: str,
                                 image_path: str,
                                 output_path: str,
                                 **kwargs) -> str:
    """
    Convenience function to generate video with subtitles.

    Args:
        audio_path: Path to audio file
        subtitles_path: Path to subtitles file
        image_path: Path to background image
        output_path: Path for output video
        **kwargs: Additional parameters

    Returns:
        Path to generated MP4 file
    """
    generator = FinalVideoGenerator()
    return generator.create_video_from_existing_files(audio_path, subtitles_path, image_path, output_path, **kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python final_video_generator.py <audio.mp3> <subtitles.srt> <image.jpg> <output.mp4>")
        sys.exit(1)

    audio_file = sys.argv[1]
    subtitles_file = sys.argv[2]
    image_file = sys.argv[3]
    output_file = sys.argv[4]

    try:
        generator = FinalVideoGenerator()
        result = generator.create_video_with_subtitles(
            audio_file, subtitles_file, image_file, output_file
        )
        print(f"Video generated successfully: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)