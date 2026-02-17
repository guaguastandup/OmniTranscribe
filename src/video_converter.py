#!/usr/bin/env python3
"""
Video to Audio Converter

This module handles conversion of video files (MP4, AVI, MOV, etc.) to audio formats (MP3, WAV)
that can be processed by the transcription engine.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import ffmpeg


class VideoConverter:
    """Convert video files to audio format for transcription."""

    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.aac', '.flac']

    def __init__(self):
        """Initialize the video converter."""
        pass

    def is_video_file(self, file_path: str) -> bool:
        """
        Check if the file is a supported video format.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a video format, False otherwise
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def is_audio_file(self, file_path: str) -> bool:
        """
        Check if the file is a supported audio format.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is an audio format, False otherwise
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_AUDIO_FORMATS

    def convert_to_audio(self,
                        video_path: str,
                        output_format: str = 'mp3',
                        output_dir: Optional[str] = None,
                        keep_original: bool = True) -> Tuple[str, bool]:
        """
        Convert video file to audio format.

        Args:
            video_path: Path to the video file
            output_format: Target audio format (mp3, wav, etc.)
            output_dir: Directory to save the converted file (default: same as video)
            keep_original: Whether to keep the original video file

        Returns:
            Tuple of (output_audio_path, was_converted)
        """
        video_path = Path(video_path)

        if not self.is_video_file(str(video_path)):
            # If it's already an audio file, return as-is
            if self.is_audio_file(str(video_path)):
                return str(video_path), False
            else:
                raise ValueError(f"Unsupported file format: {video_path.suffix}")

        # Determine output path
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            audio_path = output_dir / f"{video_path.stem}.{output_format}"
        else:
            audio_path = video_path.parent / f"{video_path.stem}.{output_format}"

        try:
            print(f"Converting video to audio: {video_path} -> {audio_path}")
            print("This may take a while depending on the video size...")

            # Use ffmpeg to convert video to audio
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec='libmp3lame' if output_format == 'mp3' else 'pcm_s16le',
                    ac=1,  # mono
                    ar='16000'  # 16kHz sample rate for better transcription
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )

            if not keep_original:
                print(f"Removing original video file: {video_path}")
                video_path.unlink()

            print(f"Successfully converted to: {audio_path}")
            return str(audio_path), True

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Video conversion failed: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Video conversion failed: {str(e)}")

    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about the video file.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video information
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_info = {}

            # Get video stream info
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']

            if video_streams:
                video_stream = video_streams[0]
                video_info.update({
                    'duration': float(video_stream.get('duration', 0)),
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'video_codec': video_stream.get('codec_name')
                })

            if audio_streams:
                audio_stream = audio_streams[0]
                video_info.update({
                    'audio_codec': audio_stream.get('codec_name'),
                    'audio_channels': audio_stream.get('channels'),
                    'audio_sample_rate': audio_stream.get('sample_rate')
                })

            return video_info

        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {str(e)}")


def convert_video_to_audio(video_path: str,
                          output_format: str = 'mp3',
                          output_dir: Optional[str] = None) -> str:
    """
    Convenience function to convert video to audio.

    Args:
        video_path: Path to the video file
        output_format: Target audio format
        output_dir: Output directory

    Returns:
        Path to the converted audio file
    """
    converter = VideoConverter()
    audio_path, was_converted = converter.convert_to_audio(
        video_path, output_format, output_dir
    )
    return audio_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_converter.py <video_file> [output_format] [output_dir]")
        sys.exit(1)

    video_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'mp3'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        converter = VideoConverter()

        if not converter.is_video_file(video_file):
            print(f"Error: {video_file} is not a supported video format")
            sys.exit(1)

        # Show video info
        print("Video Information:")
        print("=" * 40)
        info = converter.get_video_info(video_file)
        for key, value in info.items():
            print(f"{key}: {value}")
        print()

        # Convert video
        audio_path, was_converted = converter.convert_to_audio(
            video_file, output_format, output_dir
        )

        if was_converted:
            print(f"Conversion completed: {audio_path}")
        else:
            print(f"File is already audio: {audio_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)