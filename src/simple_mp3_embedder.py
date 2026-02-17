#!/usr/bin/env python3
"""
Simple MP3 Metadata Embedder using FFmpeg

This module uses FFmpeg commands to embed lyrics and cover art into MP3 files.
A simpler alternative to the complex ID3 tag handling.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict
import shutil

# PIL import with fallback
try:
    from PIL import Image
except ImportError:
    # Try to import from site-packages
    import sys
    import os
    site_packages = None

    # Try common site-packages locations
    for path in sys.path:
        if 'site-packages' in path and os.path.exists(path):
            site_packages = path
            break

    if site_packages:
        sys.path.insert(0, site_packages)
        try:
            from PIL import Image
        except ImportError:
            Image = None
    else:
        Image = None


class SimpleMP3Embedder:
    """Simple MP3 metadata embedder using FFmpeg commands."""

    def __init__(self):
        """Initialize the embedder."""
        self.supported_lyrics_formats = ['.srt', '.vtt', '.lrc']
        self.supported_image_formats = ['.jpg', '.jpeg', '.png']

    def embed_metadata(self,
                      audio_path: str,
                      lyrics_path: Optional[str] = None,
                      cover_path: Optional[str] = None,
                      title: Optional[str] = None,
                      artist: Optional[str] = None,
                      album: Optional[str] = None,
                      output_path: Optional[str] = None) -> str:
        """
        Embed metadata into MP3 file using FFmpeg.

        Args:
            audio_path: Path to input MP3 file
            lyrics_path: Path to lyrics file (SRT/VTT/LRC)
            cover_path: Path to cover image file
            title: Track title
            artist: Artist name
            album: Album name
            output_path: Output path for modified MP3 (None to create new file)

        Returns:
            Path to the MP3 file with embedded metadata
        """
        # Validate input files
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not audio_path.lower().endswith('.mp3'):
            raise ValueError("Audio file must be MP3 format")

        # Determine output path
        if output_path is None:
            base_name = Path(audio_path).stem
            output_path = f"{base_name}_with_metadata.mp3"

        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', audio_path]

        # Handle cover art first (needs to be before metadata options)
        if cover_path and os.path.exists(cover_path):
            cover_to_use = cover_path
            temp_cover = None

            # Check if we need to convert the image (only if PIL is available)
            if Image is not None:
                try:
                    # Detect image format and potentially convert
                    with Image.open(cover_path) as img:
                        # Convert to JPEG if not already JPEG
                        if img.format not in ['JPEG', 'JPG']:
                            # Create temporary JPEG file
                            temp_cover = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)

                            # Convert to RGB (remove alpha channel) and save as JPEG
                            if img.mode in ('RGBA', 'LA', 'P'):
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                                img = background

                            img.save(temp_cover.name, 'JPEG', quality=90, optimize=True)
                            cover_to_use = temp_cover.name
                except Exception as e:
                    print(f"Warning: Could not process cover image {cover_path}: {e}")
                    cover_to_use = cover_path
                    temp_cover = None
            else:
                # PIL not available, use original file
                print("Note: PIL not available, using original image file as-is")
                cover_to_use = cover_path

            cmd.extend(['-i', cover_to_use])

            # Store temp file for cleanup later
            if temp_cover:
                if not hasattr(self, '_temp_files'):
                    self._temp_files = []
                self._temp_files.append(temp_cover.name)

        # Add metadata options
        metadata_args = []

        if title:
            metadata_args.extend(['-metadata', f'title={title}'])
        if artist:
            metadata_args.extend(['-metadata', f'artist={artist}'])
        if album:
            metadata_args.extend(['-metadata', f'album={album}'])

        # Handle lyrics
        lyrics_text = None
        if lyrics_path and os.path.exists(lyrics_path):
            lyrics_text = self._process_lyrics_file(lyrics_path)
            if lyrics_text:
                # Embed lyrics as comment metadata (widest compatibility)
                metadata_args.extend(['-metadata', f'comment={lyrics_text}'])

        # Handle mapping and codec options
        if cover_path and os.path.exists(cover_path):
            # For files that might already have embedded images, we need to be more careful
            metadata_args.extend([
                '-map', '0:a',  # Map audio from first input
                '-map', '1:v',  # Map cover from second input
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-c:v', 'mjpeg',  # Force mjpeg encoding for compatibility
                '-disposition:v:0', 'attached_pic',  # Mark as attached picture
                '-write_id3v2', '1'  # Ensure ID3v2 tags are written
            ])
        else:
            # Only audio - exclude any existing video streams
            metadata_args.extend([
                '-map', '0:a',  # Map only audio from first input
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-write_id3v2', '1'  # Ensure ID3v2 tags are written
            ])

        # Add output options
        metadata_args.extend(['-y', output_path])  # Overwrite output file

        # Remove empty arguments
        metadata_args = [arg for arg in metadata_args if arg]

        # Combine command
        full_cmd = cmd + metadata_args

        try:
            print(f"Running FFmpeg command: {' '.join(full_cmd)}")
            result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)

            # Clean up temporary files
            if hasattr(self, '_temp_files'):
                for temp_file in self._temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                self._temp_files = []

            if os.path.exists(output_path):
                print(f"Successfully embedded metadata: {output_path}")
                return output_path
            else:
                raise Exception("Output file was not created")

        except subprocess.CalledProcessError as e:
            # Clean up temporary files even on error
            if hasattr(self, '_temp_files'):
                for temp_file in self._temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                self._temp_files = []
            raise Exception(f"FFmpeg failed: {e.stderr}")

    def _process_lyrics_file(self, lyrics_path: str) -> Optional[str]:
        """Process lyrics file and return text content."""
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lyrics_ext = Path(lyrics_path).suffix.lower()

            if lyrics_ext == '.lrc':
                # For LRC, we can embed it as-is with timestamps
                return content.strip()
            elif lyrics_ext in ['.srt', '.vtt']:
                # For SRT/VTT, extract just the text without timestamps
                return self._extract_lyrics_text(content)
            else:
                return content.strip()

        except Exception as e:
            print(f"Warning: Failed to process lyrics file: {e}")
            return None

    def _extract_lyrics_text(self, content: str) -> str:
        """Extract plain text from SRT or VTT format."""
        import re

        lines = content.split('\n')
        lyrics_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Skip SRT/VTT timestamp lines and metadata
            if (re.match(r'^\d+$', line) or  # SRT index
                re.match(r'^\d{2}:\d{2}:\d{2}[,\.\]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.\]\d{3}', line) or  # Timestamps
                line.startswith('WEBVTT') or  # VTT header
                re.match(r'^\d{2}:\d{2}:\d{2}[,\.\]\d{3}$', line)):  # Standalone timestamps
                continue

            # This is likely a lyrics line
            if line and not re.match(r'^[\d\s\-\[\]:.,>]+$', line):
                lyrics_lines.append(line)

        return '\n'.join(lyrics_lines)

    def get_file_info(self, audio_path: str) -> Dict:
        """Get basic information about the MP3 file using FFmpeg."""
        try:
            cmd = ['ffmpeg', '-i', audio_path, '-hide_banner']
            result = subprocess.run(cmd, capture_output=True, text=True)

            info = {
                'duration': None,
                'has_metadata': False,
                'metadata': {}
            }

            # Parse FFmpeg output for metadata
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines:
                if 'Duration:' in line:
                    # Extract duration
                    import re
                    match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                    if match:
                        hours, minutes, seconds = match.groups()
                        duration_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        info['duration'] = duration_sec

                if line.strip().startswith('metadata:'):
                    info['has_metadata'] = True
                    parts = line.strip().split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().replace('metadata:', '')
                        value = parts[1].strip()
                        info['metadata'][key] = value

            return info

        except Exception as e:
            return {'error': str(e)}


def embed_mp3_metadata(audio_path: str,
                      lyrics_path: Optional[str] = None,
                      cover_path: Optional[str] = None,
                      title: Optional[str] = None,
                      artist: Optional[str] = None,
                      album: Optional[str] = None,
                      output_path: Optional[str] = None) -> str:
    """
    Convenience function to embed metadata in MP3 file.

    Args:
        audio_path: Path to input MP3 file
        lyrics_path: Path to lyrics file (optional)
        cover_path: Path to cover image file (optional)
        title: Track title (optional)
        artist: Artist name (optional)
        album: Album name (optional)
        output_path: Output path (optional)

    Returns:
        Path to MP3 file with embedded metadata
    """
    embedder = SimpleMP3Embedder()
    return embedder.embed_metadata(
        audio_path=audio_path,
        lyrics_path=lyrics_path,
        cover_path=cover_path,
        title=title,
        artist=artist,
        album=album,
        output_path=output_path
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_mp3_embedder.py <audio.mp3> [lyrics.lrc] [cover.jpg]")
        sys.exit(1)

    audio_file = sys.argv[1]
    lyrics_file = sys.argv[2] if len(sys.argv) > 2 else None
    cover_file = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        embedder = SimpleMP3Embedder()
        result = embedder.embed_metadata(
            audio_path=audio_file,
            lyrics_path=lyrics_file,
            cover_path=cover_file
        )
        print(f"Successfully embedded metadata: {result}")

        # Show file info
        info = embedder.get_file_info(result)
        print(f"File duration: {info.get('duration', 'unknown')} seconds")
        print(f"Has metadata: {info.get('has_metadata', False)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)