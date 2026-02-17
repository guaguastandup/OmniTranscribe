#!/usr/bin/env python3
"""
OmniTranscribe - Multilingual Audio Transcription and Translation Tool

This tool processes audio/video files in any language to:
1. Convert video files to audio format (if needed)
2. Transcribe audio to subtitles using OpenAI Whisper (99+ languages)
3. Translate subtitles to any language using AI models
4. Save both original and translated subtitle files
5. Support batch processing of multiple files
6. Generate MP4 videos with synchronized subtitles from audio, lyrics, and background image
7. Embed lyrics and cover art into MP3 files for music player compatibility

Supported languages include: English, Chinese, Japanese, Korean, Spanish, French, German,
Russian, Arabic, Hindi, and 90+ more languages (auto-detection available)

Usage:
    # Simple usage (interactive mode - recommended for beginners)
    python main.py                                          # Start interactive mode
    python main.py audio_file.mp3                          # Auto-detect language, translate to Chinese

    # Specify languages
    python main.py audio_file.mp3 --language ja            # Japanese audio, auto-detect
    python main.py audio_file.mp3 --language en --target-language zh  # English to Chinese
    python main.py audio_file.mp3 --language auto          # Auto-detect source language

    # Quick commands with presets
    python main.py --fast audio_file.mp3                   # Fast processing
    python main.py --quality audio_file.mp3                # High quality
    python main.py --gpu audio_file.mp3                    # GPU accelerated
    python main.py --batch /path/to/files                  # Batch processing

    # Translation with different services (model configured in .env)
    python main.py audio_file.mp3 --translation-model gemini          # Uses GEMINI_MODEL from .env
    python main.py audio_file.mp3 --translation-model qwen            # Uses QWEN_MODEL from .env
    python main.py audio_file.mp3 --translation-model claude          # Uses CLAUDE_MODEL from .env
    python main.py audio_file.mp3 --translation-model gpt             # Uses OPENAI_MODEL from .env

    # Custom API (OpenAI compatible)
    python main.py audio_file.mp3 --translation-model custom --translation-url "https://your-api.com/v1" --translation-api-key "your-key"

    # Video generation with subtitles
    python main.py audio_file.mp3 --generate-video --background-image image.jpg

    # MP3 metadata embedding (for music players)
    python main.py audio_file.mp3 --embed-mp3-metadata --cover-image cover.jpg

    # Batch processing
    python main.py --batch /path/to/media/files --language auto --recursive

    # Other modes
    python main.py --convert-only input.srt --convert-to vtt
    python main.py --translate-only --existing-srt input.srt --target-language en
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env from the project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

from transcribe import AudioTranscriber
from translator import UniversalTranslator
from converter import SubtitleConverter
from video_converter import VideoConverter
from final_video_generator import FinalVideoGenerator
from batch_processor import BatchProcessor
from simple_mp3_embedder import SimpleMP3Embedder
from config import get_config, save_config
from interactive import run_interactive

def main():
    # Load configuration defaults
    config = get_config()

    parser = argparse.ArgumentParser(
        description="OmniTranscribe - Transcribe audio in any language and translate to any language"
    )
    parser.add_argument(
        "audio_file",
        nargs='?',  # Make audio_file optional
        help="Path to the audio/video file (mp3, wav, m4a, mp4, avi, mov, etc.)"
    )
    parser.add_argument(
        "--model",
        default=config.get('transcription.model', 'base'),
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {config.get('transcription.model', 'base')})"
    )
    parser.add_argument(
        "--device",
        default=config.get('transcription.device', 'auto'),
        choices=["auto", "cpu", "cuda", "mps"],
        help=f"Device for Whisper inference (auto, cpu, cuda, mps). Auto detects GPU if available (default: {config.get('transcription.device', 'auto')})"
    )
    parser.add_argument(
        "--language", "-l",
        default=config.get('transcription.language', 'auto'),
        help="Source language code for transcription (default: auto for auto-detection). Common: en, zh, ja, ko, es, fr, de, ru, ar, hi"
    )
    parser.add_argument(
        "--target-language", "-t",
        default=config.get('translation.target_language', 'zh'),
        help="Target language for translation (default: zh for Chinese). Use 'none' to skip translation"
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        default=config.get('transcription.strict_mode', False),
        help="Force strict transcription mode (reduces mixed language output)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode for guided configuration"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast processing preset (tiny model, larger chunks)"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use high quality preset (large model, smaller chunks)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU accelerated preset (medium model, GPU device)"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration"
    )
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )
    parser.add_argument(
        "--output-dir",
        default=config.get('output.default_dir', 'output'),
        help=f"Output directory for subtitle files (default: {config.get('output.default_dir', 'output')})"
    )
    parser.add_argument(
        "--prompt-file",
        default=config.get('translation.prompt_file', 'prompt.md'),
        help=f"Path to the translation prompt file (default: {config.get('translation.prompt_file', 'prompt.md')})"
    )
    parser.add_argument(
        "--translation-model",
        default=config.get('translation.model', 'deepseek-chat'),
        help=f"Translation model to use (default: {config.get('translation.model', 'deepseek-chat')})"
    )
    parser.add_argument(
        "--translation-api-key",
        help="API key for translation model (overrides environment variable)"
    )
    parser.add_argument(
        "--translation-url",
        help="Custom API URL for translation service (OpenAI compatible)"
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help=argparse.SUPPRESS  # Hidden for backward compatibility
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Only translate existing SRT file, skip transcription"
    )
    parser.add_argument(
        "--existing-srt",
        help="Path to existing SRT file to translate (use with --translate-only)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.get('translation.chunk_size', 20),
        help=f"Number of subtitle entries per translation chunk (default: {config.get('translation.chunk_size', 20)})"
    )
    parser.add_argument(
        "--convert-to",
        choices=["srt", "vtt", "lrc"],
        help="Convert subtitle file to specified format after processing"
    )
    parser.add_argument(
        "--convert-only",
        help="Convert existing subtitle file to specified format (e.g., --convert-only input.srt --convert-to vtt)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple files in batch mode (input_path should be a directory)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively in subdirectories for batch mode (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively in subdirectories for batch mode"
    )
    parser.add_argument(
        "--keep-video-files",
        action="store_true",
        default=True,
        help="Keep original video files after conversion (default: True)"
    )
    parser.add_argument(
        "--delete-video-files",
        action="store_true",
        help="Delete original video files after conversion"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for batch processing (default: 2)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have processed output (batch mode only)"
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate MP4 video with subtitles from audio, lyrics, and background image"
    )
    parser.add_argument(
        "--background-image",
        help="Path to background image file for video generation (use with --generate-video)"
    )
    parser.add_argument(
        "--subtitle-position",
        default="bottom",
        choices=["top", "bottom", "center"],
        help="Position of subtitles in video (default: bottom)"
    )
    parser.add_argument(
        "--subtitle-fontsize",
        type=int,
        default=48,
        help="Font size for subtitles (default: 48)"
    )
    parser.add_argument(
        "--font-path",
        help="Path to custom font file (TTF/OTF/TTC) for subtitles. Supports Chinese characters."
    )
    parser.add_argument(
        "--embed-mp3-metadata",
        action="store_true",
        help="Embed lyrics and cover art into MP3 file instead of generating MP4 video"
    )
    parser.add_argument(
        "--cover-image",
        help="Path to cover image file for MP3 metadata embedding (use with --embed-mp3-metadata)"
    )
    parser.add_argument(
        "--track-title",
        help="Track title for MP3 metadata embedding (use with --embed-mp3-metadata)"
    )
    parser.add_argument(
        "--artist-name",
        help="Artist name for MP3 metadata embedding (use with --embed-mp3-metadata)"
    )
    parser.add_argument(
        "--album-name",
        help="Album name for MP3 metadata embedding (use with --embed-mp3-metadata)"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = get_config()

    # Handle configuration commands
    if args.config:
        config.print_config()
        sys.exit(0)

    if args.reset_config:
        config.reset_to_defaults()
        print("Configuration reset to defaults.")
        sys.exit(0)

    # Handle preset options
    if args.fast:
        config.set('transcription.model', 'tiny')
        config.set('translation.chunk_size', 30)
        print("🚀 Using fast processing preset")
    elif args.quality:
        config.set('transcription.model', 'large')
        config.set('translation.chunk_size', 15)
        print("🎯 Using high quality preset")
    elif args.gpu:
        config.set('transcription.model', 'medium')
        config.set('transcription.device', 'auto')
        config.set('translation.chunk_size', 25)
        print("🔥 Using GPU accelerated preset")

    # Interactive mode
    if args.interactive or (not args.audio_file and not args.convert_only and not args.translate_only):
        try:
            result = run_interactive()
            if result.get('mode') in ['config_saved', 'config_reset']:
                sys.exit(0)
            elif result.get('mode') == 'convert':
                # Handle convert mode
                args.convert_only = result['file_path']
                args.convert_to = result['format']
            elif result.get('mode') == 'translate':
                # Handle translate mode
                args.translate_only = True
                args.existing_srt = result['file_path']
                if result.get('translation', {}).get('model'):
                    args.translation_model = result['translation']['model']
            elif result.get('mode') in ['single', 'batch']:
                # Update args with interactive results
                if result.get('mode') == 'single':
                    args.audio_file = result['file_path']
                    args.batch = False
                else:
                    args.audio_file = result['input_path']
                    args.batch = True
                    args.recursive = result.get('batch', {}).get('recursive', True)
                    args.skip_existing = result.get('batch', {}).get('skip_existing', False)

                # Apply interactive settings
                if 'transcription' in result:
                    trans = result['transcription']
                    if 'model' in trans:
                        args.model = trans['model']
                    if 'device' in trans:
                        args.device = trans['device']
                    if 'language' in trans:
                        args.language = trans['language']
                    if 'strict_mode' in trans:
                        args.strict_mode = trans['strict_mode']

                if 'translation' in result:
                    trans = result['translation']
                    if not trans.get('enable_translation', True):
                        args.target_language = 'none'
                    if 'model' in trans:
                        args.translation_model = trans['model']
                    if 'target_language' in trans:
                        args.target_language = trans['target_language']
                    if 'chunk_size' in trans:
                        args.chunk_size = trans['chunk_size']

                if 'output' in result:
                    out = result['output']
                    if 'output_dir' in out:
                        args.output_dir = out['output_dir']
                    if 'format' in out:
                        args.convert_to = out['format']

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    # Resolve conflicting arguments
    if args.no_recursive:
        args.recursive = False

    keep_video_files = args.keep_video_files and not args.delete_video_files

    # Validate video generation arguments
    if args.generate_video and not args.background_image:
        print("Error: --background-image is required when using --generate-video")
        sys.exit(1)

    if args.generate_video and args.batch and not os.path.exists(args.background_image):
        print(f"Error: Background image not found: {args.background_image}")
        sys.exit(1)

    if args.generate_video and not args.translate_only and not args.audio_file:
        print("Error: audio_file is required when using --generate-video without --translate-only")
        sys.exit(1)

    if args.generate_video and args.translate_only and not args.audio_file:
        print("Error: audio_file is required when using --generate-video with --translate-only")
        sys.exit(1)

    # Validate MP3 metadata embedding arguments
    if args.embed_mp3_metadata and args.generate_video:
        print("Error: Cannot use --embed-mp3-metadata and --generate-video together")
        sys.exit(1)

    if args.embed_mp3_metadata and args.cover_image and not os.path.exists(args.cover_image):
        print(f"Error: Cover image not found: {args.cover_image}")
        sys.exit(1)

    # Validate arguments
    if args.convert_only:
        # Convert-only mode
        if not args.convert_to:
            print("Error: --convert-to is required when using --convert-only")
            sys.exit(1)

        if not os.path.exists(args.convert_only):
            print(f"Error: Input file not found: {args.convert_only}")
            sys.exit(1)

        # Perform conversion only
        print("Format conversion mode")
        print("=" * 50)

        try:
            converter = SubtitleConverter()
            output_file = converter.convert_file(args.convert_only, args.convert_to)
            print(f"Successfully converted to: {output_file}")
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            sys.exit(1)

        return

    # Batch processing mode
    if args.batch:
        if not args.audio_file:
            print("Error: input_path is required when using --batch")
            sys.exit(1)

        if not os.path.exists(args.audio_file):
            print(f"Error: Input path not found: {args.audio_file}")
            sys.exit(1)

        print("Batch processing mode")
        print("=" * 50)
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        print("=" * 50)

        try:
            processor = BatchProcessor(
                model_size=args.model,
                device=args.device,
                deepseek_model=args.translation_model,
                chunk_size=args.chunk_size,
                prompt_file=args.prompt_file,
                max_workers=args.max_workers,
                generate_video=args.generate_video,
                background_image=args.background_image,
                subtitle_position=args.subtitle_position,
                subtitle_fontsize=args.subtitle_fontsize,
                font_path=args.font_path
            )

            # Progress callback
            def progress_callback(completed, total, result):
                status_symbol = {
                    'completed': '✓',
                    'error': '✗',
                    'skipped': '-',
                    'already_exists': '⊘'
                }.get(result['status'], '?')

                print(f"[{completed}/{total}] {status_symbol} {Path(result['input_file']).name}")

            # Process files
            results = processor.process_batch(
                input_path=args.audio_file,
                output_dir=args.output_dir,
                recursive=args.recursive,
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

        except Exception as e:
            print(f"Batch processing failed: {str(e)}")
            sys.exit(1)

        return

    # Regular processing mode
    if not args.audio_file and not args.translate_only:
        print("Error: audio_file is required when not using --convert-only, --batch, or --translate-only")
        sys.exit(1)

    if not args.translate_only and not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    if args.translate_only and not args.existing_srt:
        print("Error: --existing-srt is required when using --translate-only")
        sys.exit(1)

    if args.translate_only and not os.path.exists(args.existing_srt):
        print(f"Error: SRT file not found: {args.existing_srt}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate base filename
    if args.translate_only:
        # For translate-only mode, use the existing SRT file's stem but ensure it's in the output directory
        base_name = Path(args.existing_srt).stem
        current_audio_path = args.audio_file if args.audio_file else None  # Use the provided audio file (can be None if not provided)
    else:
        base_name = Path(args.audio_file).stem
        current_audio_path = args.audio_file

    original_srt_path = output_dir / f"{base_name}_original.srt"
    translated_srt_path = output_dir / f"{base_name}_translated.srt"

    try:
        # Step 1: Convert video to audio if needed (unless translate-only)
        if not args.translate_only:
            video_converter = VideoConverter()

            if video_converter.is_video_file(current_audio_path):
                print("=" * 50)
                print("Step 1: Converting video to audio")
                print("=" * 50)

                current_audio_path, was_converted = video_converter.convert_to_audio(
                    current_audio_path,
                    output_format='mp3',
                    output_dir=str(output_dir),
                    keep_original=keep_video_files
                )

                if was_converted:
                    print(f"Video converted to audio: {current_audio_path}")

                print()

            # Step 2: Transcribe audio to subtitles
            print("=" * 50)
            source_lang = args.language if args.language != 'auto' else 'auto-detected'
            print(f"Step 2: Transcribing audio ({source_lang}) to subtitles")
            print("=" * 50)

            transcriber = AudioTranscriber(model_size=args.model, device=args.device)

            # Configure transcription language and mode
            if args.strict_mode:
                print(f"Using strict transcription mode for: {args.language}")
                srt_content = transcriber.transcribe_audio(
                    current_audio_path,
                    language=args.language,
                    progress_callback=lambda current, total: None  # Simple callback
                )
            else:
                srt_content = transcriber.transcribe_audio(
                    current_audio_path,
                    language=args.language,
                    progress_callback=lambda current, total: None  # Simple callback
                )
            transcriber.save_srt(srt_content, str(original_srt_path))

        else:
            print("=" * 50)
            print("Using existing SRT file for translation")
            print("=" * 50)

            with open(args.existing_srt, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            # Also copy the existing file to output directory for consistency
            with open(original_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

        # Step 3: Translate to target language
        if args.target_language != 'none':
            print("\n" + "=" * 50)
            print(f"Step 3: Translating subtitles to {args.target_language}")
            print("=" * 50)

        # Configure translator based on arguments
        if args.translation_model == "custom":
            if not args.translation_url or not args.translation_api_key:
                print("Error: --translation-url and --translation-api-key are required when using custom model")
                sys.exit(1)
            translator = UniversalTranslator(
                api_key=args.translation_api_key,
                model=args.translation_model,
                base_url=args.translation_url
            )
        else:
            translator = UniversalTranslator(
                api_key=args.translation_api_key,
                model=args.translation_model,
                base_url=args.translation_url
            )

        print(f"Using translation model: {args.translation_model}")
        if args.translation_url:
            print(f"Using custom API URL: {args.translation_url}")

        # Only translate if target_language is not 'none'
        if args.target_language != 'none':
            translated_content = translator.translate_srt(
                srt_content,
                prompt_path=args.prompt_file,
                chunk_size=args.chunk_size,
                target_language=args.target_language
            )
            translator.save_translated_srt(translated_content, str(translated_srt_path))

        # Step 3: Summary
        print("\n" + "=" * 50)
        print("Processing completed successfully!")
        print("=" * 50)
        if not args.translate_only:
            print(f"Original subtitles: {original_srt_path}")
        else:
            print(f"Original subtitles copied to: {original_srt_path}")
        if args.target_language != 'none':
            print(f"Translated subtitles ({args.target_language}): {translated_srt_path}")

        # Step 4: Format conversion (if requested)
        if args.convert_to:
            print("\n" + "=" * 50)
            print("Step 4: Converting subtitle format")
            print("=" * 50)

            try:
                converter = SubtitleConverter()

                # Convert original subtitles (if not translate-only)
                if not args.translate_only and args.convert_to != 'srt':
                    converted_original = converter.convert_file(
                        str(original_srt_path), args.convert_to
                    )
                    print(f"Original subtitles converted to: {converted_original}")

                # Convert translated subtitles (if different format)
                if args.convert_to != 'srt':
                    converted_translated = converter.convert_file(
                        str(translated_srt_path), args.convert_to
                    )
                    print(f"Translated subtitles converted to: {converted_translated}")

                print("Format conversion completed successfully!")

            except Exception as e:
                print(f"Format conversion failed: {str(e)}")
                print("You can manually convert the files later using the converter module.")

        # Step 5: Video generation or MP3 metadata embedding (if requested)
        if args.generate_video:
            print("\n" + "=" * 50)
            print("Step 5: Generating MP4 video with subtitles")
            print("=" * 50)

            try:
                # Check if background image exists
                if not os.path.exists(args.background_image):
                    print(f"Error: Background image not found: {args.background_image}")
                    sys.exit(1)

                # Generate output path for video
                video_output_path = output_dir / f"{base_name}_video.mp4"

                # Handle lyrics file for video generation
                converter = SubtitleConverter()
                if args.translate_only and args.existing_srt:
                    # For translate-only mode, we need to check if the existing file is SRT or needs conversion
                    if args.existing_srt.lower().endswith('.srt'):
                        # Convert the existing SRT or use the translated SRT
                        if os.path.exists(translated_srt_path):
                            # convert_file returns the actual output path
                            lyrics_file = converter.convert_file(str(translated_srt_path), 'lrc')
                        else:
                            # Use the existing SRT file and convert it to LRC
                            lyrics_file = converter.convert_file(args.existing_srt, 'lrc')
                    else:
                        # Use the existing file directly if it's already in LRC format
                        lyrics_file = args.existing_srt
                else:
                    # Convert translated SRT to LRC
                    # convert_file returns the actual output path
                    lyrics_file = converter.convert_file(str(translated_srt_path), 'lrc')

                # Create video with subtitles
                video_generator = FinalVideoGenerator(custom_font_path=args.font_path)
                result_path = video_generator.create_video_from_existing_files(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    image_path=args.background_image,
                    output_path=str(video_output_path)
                )

                print(f"Video generation completed successfully!")
                print(f"Output video: {result_path}")

            except Exception as e:
                print(f"Video generation failed: {str(e)}")
                print("You can try generating the video manually using the video_generator module.")

        # Step 6: MP3 metadata embedding (if requested)
        if args.embed_mp3_metadata:
            print("\n" + "=" * 50)
            print("Step 6: Embedding lyrics and cover art into MP3")
            print("=" * 50)

            try:
                # Ensure we have an MP3 file to work with
                if not current_audio_path.lower().endswith('.mp3'):
                    print("Error: MP3 metadata embedding requires an MP3 audio file")
                    sys.exit(1)

                # Handle lyrics file for metadata embedding
                if args.translate_only and args.existing_srt:
                    # For translate-only mode, use the existing lyrics file
                    lyrics_file = args.existing_srt
                else:
                    # Use the translated SRT file (will be processed by embedder)
                    lyrics_file = str(translated_srt_path)

                # Generate output path for MP3 with metadata
                mp3_output_path = output_dir / f"{base_name}_with_metadata.mp3"

                # Create MP3 metadata embedder
                embedder = SimpleMP3Embedder()
                result_path = embedder.embed_metadata(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    cover_path=args.cover_image,
                    title=args.track_title,
                    artist=args.artist_name,
                    album=args.album_name,
                    output_path=str(mp3_output_path)
                )

                print(f"MP3 metadata embedding completed successfully!")
                print(f"Output MP3: {result_path}")

                # Show embedded metadata info
                info = embedder.get_file_info(result_path)
                print(f"File duration: {info.get('duration', 'unknown')} seconds")
                print(f"Has metadata: {info.get('has_metadata', False)}")

            except Exception as e:
                print(f"MP3 metadata embedding failed: {str(e)}")
                print("You can try embedding metadata manually using the mp3_metadata_embedder module.")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()