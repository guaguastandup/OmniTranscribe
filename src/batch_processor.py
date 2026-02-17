#!/usr/bin/env python3
"""
Batch Audio Processor

This module handles batch processing of multiple audio/video files for transcription
and translation. It supports recursive directory traversal and parallel processing.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Callable
# from concurrent.futures import ThreadPoolExecutor, as_completed  # No longer needed for sequential processing
import time

from transcribe import AudioTranscriber
from translator import UniversalTranslator, get_prompt_path
from converter import SubtitleConverter
from video_converter import VideoConverter
from final_video_generator import FinalVideoGenerator


class BatchProcessor:
    """Process multiple audio/video files in batch."""

    def __init__(self,
                 model_size: str = "base",
                 device: str = "auto",
                 deepseek_model: str = "deepseek-chat",
                 chunk_size: int = 20,
                 prompt_file: str = "prompt.md",
                 max_workers: int = 2,
                 generate_video: bool = False,
                 background_image: Optional[str] = None,
                 subtitle_position: str = "bottom",
                 subtitle_fontsize: int = 48,
                 font_path: Optional[str] = None):
        """
        Initialize the batch processor.

        Args:
            model_size: Whisper model size
            device: Device for Whisper inference (auto, cpu, cuda, mps)
            deepseek_model: DeepSeek model for translation
            chunk_size: Number of subtitle entries per translation chunk
            prompt_file: Path to translation prompt file
            max_workers: Maximum number of parallel workers
            generate_video: Whether to generate MP4 videos with subtitles
            background_image: Path to background image for video generation
            subtitle_position: Position of subtitles in video (top, bottom, center)
            subtitle_fontsize: Font size for subtitles in video
            font_path: Path to custom font file for subtitles (TTF/OTF/TTC)
        """
        self.transcriber = AudioTranscriber(model_size, device)
        self.translator = UniversalTranslator(model=deepseek_model)
        self.converter = SubtitleConverter()
        self.video_converter = VideoConverter()

        self.translation_model = deepseek_model
        self.chunk_size = chunk_size
        self.prompt_file = get_prompt_path(prompt_file)
        self.max_workers = max_workers

        # Video generation settings
        self.generate_video = generate_video
        self.background_image = background_image
        self.subtitle_position = subtitle_position
        self.subtitle_fontsize = subtitle_fontsize
        self.font_path = font_path

    def find_media_files(self,
                        input_path: str,
                        recursive: bool = True,
                        supported_formats: Optional[List[str]] = None) -> List[str]:
        """
        Find all supported audio/video files in the given path.

        Args:
            input_path: Path to file or directory
            recursive: Whether to search recursively in subdirectories
            supported_formats: List of supported file extensions

        Returns:
            List of file paths to process
        """
        if supported_formats is None:
            supported_formats = (
                VideoConverter.SUPPORTED_VIDEO_FORMATS +
                VideoConverter.SUPPORTED_AUDIO_FORMATS
            )

        input_path = Path(input_path)
        media_files = []

        if input_path.is_file():
            # Single file mode
            if input_path.suffix.lower() in supported_formats:
                media_files.append(str(input_path))
        elif input_path.is_dir():
            # Directory mode
            pattern = "**/*" if recursive else "*"
            for ext in supported_formats:
                media_files.extend(
                    str(f) for f in input_path.glob(f"{pattern}{ext}")
                )
                media_files.extend(
                    str(f) for f in input_path.glob(f"{pattern}{ext.upper()}")
                )

        return sorted(media_files)

    def process_single_file(self,
                           file_path: str,
                           output_dir: str,
                           convert_to: Optional[str] = None,
                           keep_video_files: bool = True,
                           skip_existing: bool = False) -> Dict[str, str]:
        """
        Process a single audio/video file.

        Args:
            file_path: Path to the media file
            output_dir: Output directory for results
            convert_to: Output subtitle format (srt, vtt, lrc)
            keep_video_files: Whether to keep original video files after conversion
            skip_existing: Skip files that already have processed output

        Returns:
            Dictionary containing result file paths
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create subdirectory for this file to organize outputs
        file_output_dir = output_dir / file_path.stem
        file_output_dir.mkdir(exist_ok=True)

        base_name = file_path.stem
        original_srt_path = file_output_dir / f"{base_name}_original.srt"
        translated_srt_path = file_output_dir / f"{base_name}_translated.srt"

        result = {
            'input_file': str(file_path),
            'status': 'skipped',
            'original_srt': str(original_srt_path),
            'translated_srt': str(translated_srt_path),
            'error': None
        }

        # Check if files already exist
        if skip_existing and original_srt_path.exists() and translated_srt_path.exists():
            result['status'] = 'already_exists'
            return result

        try:
            current_audio_path = str(file_path)

            # Step 1: Convert video to audio if needed
            if self.video_converter.is_video_file(current_audio_path):
                print(f"[{file_path.name}] Converting video to audio...")
                current_audio_path, was_converted = self.video_converter.convert_to_audio(
                    current_audio_path,
                    output_format='mp3',
                    output_dir=str(file_output_dir),
                    keep_original=keep_video_files
                )

                if was_converted:
                    result['converted_audio'] = current_audio_path

            # Step 2: Transcribe audio
            print(f"[{file_path.name}] Transcribing audio...")
            srt_content = self.transcriber.transcribe_audio(current_audio_path)
            self.transcriber.save_srt(srt_content, str(original_srt_path))

            # Step 3: Translate subtitles
            print(f"[{file_path.name}] Translating subtitles...")
            translated_content = self.translator.translate_srt(
                srt_content,
                prompt_path=self.prompt_file,
                chunk_size=self.chunk_size
            )
            self.translator.save_translated_srt(translated_content, str(translated_srt_path))

            # Step 4: Convert subtitle format if requested
            if convert_to and convert_to != 'srt':
                print(f"[{file_path.name}] Converting subtitle format to {convert_to}...")

                # Convert original subtitles
                if convert_to != 'srt':
                    converted_original = self.converter.convert_file(
                        str(original_srt_path), convert_to
                    )
                    result[f'original_{convert_to}'] = converted_original

                # Convert translated subtitles
                converted_translated = self.converter.convert_file(
                    str(translated_srt_path), convert_to
                )
                result[f'translated_{convert_to}'] = converted_translated

            # Step 5: Generate video if requested
            if self.generate_video:
                if not self.background_image:
                    raise ValueError("Background image is required for video generation")

                print(f"[{file_path.name}] Generating MP4 video...")

                # Convert translated SRT to LRC for video
                # convert_file returns the actual output path
                lyrics_file = self.converter.convert_file(str(translated_srt_path), 'lrc')

                # Generate output path for video
                video_output_path = file_output_dir / f"{base_name}_video.mp4"

                # Create video with subtitles
                video_generator = FinalVideoGenerator(custom_font_path=self.font_path)
                result_video_path = video_generator.create_video_from_existing_files(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    image_path=self.background_image,
                    output_path=str(video_output_path)
                )

                result['video_file'] = result_video_path

            result['status'] = 'completed'

            # Clean up temporary audio file if it was converted from video
            if 'converted_audio' in result and not keep_video_files:
                try:
                    os.unlink(result['converted_audio'])
                except:
                    pass

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"[{file_path.name}] Error: {str(e)}")

        return result

    def process_batch(self,
                     input_path: str,
                     output_dir: str,
                     recursive: bool = True,
                     convert_to: Optional[str] = None,
                     keep_video_files: bool = True,
                     skip_existing: bool = False,
                     progress_callback: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Process multiple files in batch.

        Args:
            input_path: Path to file or directory to process
            output_dir: Output directory for results
            recursive: Whether to search recursively in subdirectories
            convert_to: Output subtitle format (srt, vtt, lrc)
            keep_video_files: Whether to keep original video files
            skip_existing: Skip files that already have processed output
            progress_callback: Function to call with progress updates

        Returns:
            List of processing results for each file
        """
        # Find all media files
        media_files = self.find_media_files(input_path, recursive)

        if not media_files:
            print(f"No supported media files found in: {input_path}")
            return []

        print(f"Found {len(media_files)} files to process")
        print("=" * 50)

        results = []
        completed = 0

        # Process files sequentially (not in parallel)
        for file_path in media_files:
            result = self.process_single_file(
                file_path,
                output_dir,
                convert_to,
                keep_video_files,
                skip_existing
            )
            results.append(result)
            completed += 1

            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed, len(media_files), result)

            # Print progress
            status_symbol = {
                'completed': '✓',
                'error': '✗',
                'skipped': '-',
                'already_exists': '⊘'
            }.get(result['status'], '?')

            print(f"[{completed}/{len(media_files)}] {status_symbol} {Path(result['input_file']).name}")

        return results

    def generate_summary_report(self, results: List[Dict[str, str]]) -> str:
        """
        Generate a summary report of the batch processing.

        Args:
            results: List of processing results

        Returns:
            Formatted summary report
        """
        total = len(results)
        completed = sum(1 for r in results if r['status'] == 'completed')
        errors = sum(1 for r in results if r['status'] == 'error')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        already_exists = sum(1 for r in results if r['status'] == 'already_exists')

        report = []
        report.append("=" * 60)
        report.append("BATCH PROCESSING SUMMARY")
        report.append("=" * 60)
        report.append(f"Total files processed: {total}")
        report.append(f"Successfully completed: {completed}")
        report.append(f"Already existed (skipped): {already_exists}")
        report.append(f"Errors: {errors}")
        report.append(f"Skipped: {skipped}")
        report.append("")

        if errors > 0:
            report.append("ERRORS:")
            report.append("-" * 40)
            for result in results:
                if result['status'] == 'error':
                    filename = Path(result['input_file']).name
                    report.append(f"• {filename}: {result['error']}")
            report.append("")

        return "\n".join(report)


def main():
    """Command line interface for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process audio/video files for transcription and translation"
    )
    parser.add_argument(
        "input_path",
        help="Path to media file or directory containing media files"
    )
    parser.add_argument(
        "--output-dir",
        default="batch_output",
        help="Output directory for processed files (default: batch_output)"
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for Whisper inference (auto, cpu, cuda, mps). Auto detects GPU if available (default: auto)"
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help="DeepSeek model for translation (default: deepseek-chat)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of subtitle entries per translation chunk (default: 20)"
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.md",
        help="Path to translation prompt file (default: prompt.md)"
    )
    parser.add_argument(
        "--convert-to",
        choices=["srt", "vtt", "lrc"],
        help="Convert subtitles to specified format"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--keep-video-files",
        action="store_true",
        default=True,
        help="Keep original video files after conversion"
    )
    parser.add_argument(
        "--delete-video-files",
        action="store_true",
        help="Delete original video files after conversion (overrides --keep-video-files)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively in subdirectories"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have processed output"
    )

    args = parser.parse_args()

    # Determine video file handling
    keep_video_files = args.keep_video_files and not args.delete_video_files

    # Create batch processor
    processor = BatchProcessor(
        model_size=args.model,
        device=args.device,
        deepseek_model=args.deepseek_model,
        chunk_size=args.chunk_size,
        prompt_file=args.prompt_file,
        max_workers=args.max_workers
    )

    # Progress callback
    def progress_callback(completed, total, result):
        pass  # Simple progress printing is handled in the main function

    print("Starting batch processing...")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.max_workers}")
    print("=" * 50)

    # Process files
    results = processor.process_batch(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        convert_to=args.convert_to,
        keep_video_files=keep_video_files,
        skip_existing=args.skip_existing,
        progress_callback=progress_callback
    )

    # Print summary
    print("\n" + processor.generate_summary_report(results))

    # Return with error code if there were any errors
    if any(r['status'] == 'error' for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()