#!/usr/bin/env python3
"""
Subtitle format converter module

Supports conversion between:
- SRT (SubRip)
- VTT (WebVTT)
- LRC (Lyrics)
"""

import re
import os
from typing import List, Tuple
from datetime import timedelta

class SubtitleConverter:
    def __init__(self):
        """Initialize the subtitle converter"""
        pass

    def parse_srt(self, srt_content: str) -> List[Tuple[int, float, float, str]]:
        """
        Parse SRT content into structured data

        Args:
            srt_content: SRT format content

        Returns:
            List of tuples: (index, start_time, end_time, text)
        """
        subtitles = []

        # Split by double newlines to get subtitle blocks
        blocks = re.split(r'\n\s*\n', srt_content.strip())

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse index
                index = int(lines[0].strip())

                # Parse time line
                time_line = lines[1].strip()
                match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
                if not match:
                    continue

                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, match.groups())
                start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

                # Parse text (may be multiple lines)
                text = '\n'.join(lines[2:]).strip()

                subtitles.append((index, start_time, end_time, text))

            except (ValueError, IndexError):
                continue

        return subtitles

    def srt_to_vtt(self, srt_content: str) -> str:
        """
        Convert SRT format to WebVTT format

        Args:
            srt_content: SRT format content

        Returns:
            WebVTT format content
        """
        subtitles = self.parse_srt(srt_content)

        vtt_lines = ["WEBVTT", ""]

        for index, start_time, end_time, text in subtitles:
            # Convert to VTT time format
            start_vtt = self._seconds_to_vtt_time(start_time)
            end_vtt = self._seconds_to_vtt_time(end_time)

            vtt_lines.append(f"{index}")
            vtt_lines.append(f"{start_vtt} --> {end_vtt}")
            vtt_lines.append(text)
            vtt_lines.append("")  # Empty line between entries

        return '\n'.join(vtt_lines)

    def srt_to_lrc(self, srt_content: str) -> str:
        """
        Convert SRT format to LRC (lyrics) format

        Args:
            srt_content: SRT format content

        Returns:
            LRC format content
        """
        subtitles = self.parse_srt(srt_content)

        lrc_lines = []

        for index, start_time, end_time, text in subtitles:
            # Convert to LRC time format [mm:ss.xx]
            lrc_time = self._seconds_to_lrc_time(start_time)

            # Clean text for LRC format (remove newlines, combine into single line)
            clean_text = text.replace('\n', ' ').strip()

            # Add metadata for LRC
            lrc_lines.append(f"[{lrc_time}] {clean_text}")

        return '\n'.join(lrc_lines)

    def vtt_to_srt(self, vtt_content: str) -> str:
        """
        Convert WebVTT format to SRT format

        Args:
            vtt_content: WebVTT format content

        Returns:
            SRT format content
        """
        lines = vtt_content.split('\n')
        subtitles = []
        current_subtitle = []
        index = 1

        # Skip WEBVTT header and empty lines
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('WEBVTT'):
                start_idx = i + 1
                break

        for line in lines[start_idx:]:
            line = line.strip()

            if not line:
                if current_subtitle:
                    # Process completed subtitle
                    if len(current_subtitle) >= 2:
                        time_line = current_subtitle[0]
                        text_lines = current_subtitle[1:]

                        # Skip if this is just a number (cue identifier)
                        if time_line.isdigit() and len(current_subtitle) >= 3:
                            time_line = current_subtitle[1]
                            text_lines = current_subtitle[2:]

                        # Parse VTT time
                        match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})', time_line)
                        if match:
                            start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, match.groups())
                            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

                            # Convert to SRT format
                            start_srt = self._seconds_to_srt_time(start_time)
                            end_srt = self._seconds_to_srt_time(end_time)
                            text = '\n'.join(text_lines).strip()

                            if text:  # Only add if there's actual text
                                subtitles.append((index, start_srt, end_srt, text))
                                index += 1

                    current_subtitle = []
            else:
                current_subtitle.append(line)

        # Generate SRT content
        srt_lines = []
        for idx, start_time, end_time, text in subtitles:
            srt_lines.append(str(idx))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries

        return '\n'.join(srt_lines)

    def lrc_to_srt(self, lrc_content: str) -> str:
        """
        Convert LRC format to SRT format

        Args:
            lrc_content: LRC format content

        Returns:
            SRT format content
        """
        lines = lrc_content.split('\n')
        subtitles = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('[offset:') or line.startswith('[ti:') or line.startswith('[ar:'):
                continue

            # Parse LRC timestamp [mm:ss.xx]
            match = re.match(r'\[(\d{2}):(\d{2})(?:\.(\d{2}))?\](.*)', line)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                centiseconds = int(match.group(3) or '00')
                text = match.group(4).strip()

                start_time = minutes * 60 + seconds + centiseconds / 100

                # Estimate end time (next subtitle start or +3 seconds as default)
                end_time = start_time + 3.0  # Default duration

                # Convert to SRT format
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)

                subtitles.append((len(subtitles) + 1, start_srt, end_srt, text))

        # Generate SRT content
        srt_lines = []
        for idx, start_time, end_time, text in subtitles:
            srt_lines.append(str(idx))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries

        return '\n'.join(srt_lines)

    def convert_file(self, input_file: str, output_format: str) -> str:
        """
        Convert subtitle file to specified format

        Args:
            input_file: Path to input subtitle file
            output_format: Target format (srt, vtt, lrc)

        Returns:
            Path to output file
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Detect input format from file extension
        input_ext = os.path.splitext(input_file)[1].lower()

        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert based on input and output formats
        output_content = ""
        if input_ext == '.srt':
            if output_format.lower() == 'vtt':
                output_content = self.srt_to_vtt(content)
            elif output_format.lower() == 'lrc':
                output_content = self.srt_to_lrc(content)
            elif output_format.lower() == 'srt':
                output_content = content
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        elif input_ext == '.vtt':
            if output_format.lower() == 'srt':
                output_content = self.vtt_to_srt(content)
            elif output_format.lower() == 'vtt':
                output_content = content
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        elif input_ext == '.lrc':
            if output_format.lower() == 'srt':
                output_content = self.lrc_to_srt(content)
            elif output_format.lower() == 'lrc':
                output_content = content
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        else:
            raise ValueError(f"Unsupported input format: {input_ext}")

        # Generate output file path
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.{output_format.lower()}"

        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)

        return output_file

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def _seconds_to_lrc_time(self, seconds: float) -> str:
        """Convert seconds to LRC time format (mm:ss.xx)"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)

        return f"{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"