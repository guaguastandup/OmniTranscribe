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

from .translator import UniversalTranslator
from .converter import SubtitleConverter
from .final_video_generator import FinalVideoGenerator
from .batch_processor import BatchProcessor
from .simple_mp3_embedder import SimpleMP3Embedder
from .config import get_config
from .interactive import run_interactive

# Import cache module
try:
    from .cache import get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


def _run_translate_only(args, output_dir, keep_video_files):
    """Translate an existing SRT file and optionally produce exports from it.

    The transcription stages are skipped; the existing file is used as the
    source for translation, format conversion, video generation and MP3
    metadata embedding.
    """
    base_name = Path(args.existing_srt).stem
    current_audio_path = args.audio_file

    original_srt_path = output_dir / f"{base_name}_original.srt"
    translated_srt_path = output_dir / f"{base_name}_translated.srt"

    try:
        print("=" * 50)
        print("Using existing SRT file for translation")
        print("=" * 50)

        with open(args.existing_srt, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        # Copy the existing file to the output directory for consistency
        with open(original_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        # Translate to target language (translator is only built when
        # translation is actually requested)
        if args.target_language != 'none':
            print("\n" + "=" * 50)
            print(f"Translating subtitles to {args.target_language}")
            print("=" * 50)

            if args.translation_model == "custom" and (not args.translation_url or not args.translation_api_key):
                print("Error: --translation-url and --translation-api-key are required when using custom model")
                sys.exit(1)

            translator = UniversalTranslator(
                api_key=args.translation_api_key,
                model=args.translation_model,
                base_url=args.translation_url
            )

            print(f"Using translation model: {args.translation_model}")
            translated_content = translator.translate_srt(
                srt_content,
                prompt_path=args.prompt_file,
                chunk_size=args.chunk_size,
                target_language=args.target_language
            )
            translator.save_translated_srt(translated_content, str(translated_srt_path))

        print("\n" + "=" * 50)
        print("Processing completed successfully!")
        print("=" * 50)
        print(f"Original subtitles copied to: {original_srt_path}")
        if args.target_language != 'none':
            print(f"Translated subtitles ({args.target_language}): {translated_srt_path}")

        # Format conversion (if requested)
        if args.convert_to:
            print("\n" + "=" * 50)
            print("Converting subtitle format")
            print("=" * 50)
            try:
                converter = SubtitleConverter()
                if args.convert_to != 'srt' and args.target_language != 'none':
                    converted = converter.convert_file(str(translated_srt_path), args.convert_to)
                    print(f"Translated subtitles converted to: {converted}")
                print("Format conversion completed successfully!")
            except Exception as e:
                print(f"Format conversion failed: {str(e)}")

        # Video generation (if requested)
        if args.generate_video:
            print("\n" + "=" * 50)
            print("Generating MP4 video with subtitles")
            print("=" * 50)
            try:
                if not os.path.exists(args.background_image):
                    print(f"Error: Background image not found: {args.background_image}")
                    sys.exit(1)

                video_output_path = output_dir / f"{base_name}_video.mp4"

                # Non-SRT lyrics files (e.g. LRC) can be used directly
                if not args.existing_srt.lower().endswith('.srt'):
                    lyrics_file = args.existing_srt
                elif args.target_language != 'none':
                    lyrics_file = SubtitleConverter().convert_file(str(translated_srt_path), 'lrc')
                else:
                    lyrics_file = SubtitleConverter().convert_file(str(original_srt_path), 'lrc')

                video_generator = FinalVideoGenerator(custom_font_path=args.font_path)
                result_path = video_generator.create_video_from_existing_files(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    image_path=args.background_image,
                    output_path=str(video_output_path),
                    font_size=args.subtitle_fontsize,
                    subtitle_position=args.subtitle_position
                )
                print("Video generation completed successfully!")
                print(f"Output video: {result_path}")
            except Exception as e:
                print(f"Video generation failed: {str(e)}")

        # MP3 metadata embedding (if requested)
        if args.embed_mp3_metadata:
            print("\n" + "=" * 50)
            print("Embedding lyrics and cover art into MP3")
            print("=" * 50)
            try:
                if not current_audio_path.lower().endswith('.mp3'):
                    print("Error: MP3 metadata embedding requires an MP3 audio file")
                    sys.exit(1)

                mp3_output_path = output_dir / f"{base_name}_with_metadata.mp3"
                embedder = SimpleMP3Embedder()
                result_path = embedder.embed_metadata(
                    audio_path=current_audio_path,
                    lyrics_path=args.existing_srt,
                    cover_path=args.cover_image,
                    title=args.track_title,
                    artist=args.artist_name,
                    album=args.album_name,
                    output_path=str(mp3_output_path)
                )
                print("MP3 metadata embedding completed successfully!")
                print(f"Output MP3: {result_path}")
            except Exception as e:
                print(f"MP3 metadata embedding failed: {str(e)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

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
        default=os.getenv('WHISPER_MODEL', config.get('transcription.model', 'base')),
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (env: WHISPER_MODEL)"
    )
    parser.add_argument(
        "--device",
        default=config.get('transcription.device', 'auto'),
        choices=["auto", "cpu", "cuda", "mps"],
        help=f"Device for Whisper inference (auto, cpu, cuda, mps). Auto detects GPU if available (default: {config.get('transcription.device', 'auto')})"
    )
    parser.add_argument(
        "--language", "-l",
        default=os.getenv('WHISPER_LANGUAGE', config.get('transcription.language', 'auto')),
        help="Source language code for transcription (env: WHISPER_LANGUAGE; default: auto for auto-detection). Common: en, zh, ja, ko, es, fr, de, ru, ar, hi"
    )
    parser.add_argument(
        "--target-language", "-t",
        default=os.getenv('WHISPER_TARGET_LANGUAGE', config.get('translation.target_language', 'zh')),
        help="Target language for translation (env: WHISPER_TARGET_LANGUAGE; default: zh). Use 'none' to skip translation"
    )
    parser.add_argument(
        "--initial-prompt",
        help="Transcription hint for Whisper: proper nouns, terminology, names (improves domain-specific accuracy)"
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
        default=os.getenv('TRANSLATION_MODEL', config.get('translation.model', 'deepseek-chat')),
        help="Translation service to use: google, deepseek, gemini, qwen, claude, gpt, custom (env: TRANSLATION_MODEL)"
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
        help="Reserved for future parallel batch processing; files are currently processed sequentially (default: 2)"
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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable transcription cache (will re-transcribe even if cached result exists)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached transcriptions and exit"
    )
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics and exit"
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

    # Handle cache commands
    if args.clear_cache:
        if CACHE_AVAILABLE:
            cache = get_cache()
            cache.clear()
            print("All cached transcriptions have been cleared.")
        else:
            print("Cache module not available.")
        sys.exit(0)

    if args.cache_stats:
        if CACHE_AVAILABLE:
            cache = get_cache()
            stats = cache.get_cache_stats()
            print("Cache Statistics:")
            print("=" * 40)
            print(f"Total entries: {stats['total_entries']}")
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Cache directory: {stats['cache_dir']}")

            # List cached items
            cached = cache.list_cached()
            if cached:
                print("\nCached items:")
                for item in cached:
                    print(f"  - {item['filename']} ({item['model']}, {item['language']}) - {item['timestamp']}")
            else:
                print("\nNo cached transcriptions.")
        else:
            print("Cache module not available.")
        sys.exit(0)

    # Handle preset options (must be applied to args directly: argparse defaults
    # were already resolved when the parser was built)
    if args.fast:
        args.model = 'tiny'
        args.chunk_size = 30
        print("🚀 Using fast processing preset")
    elif args.quality:
        args.model = 'large'
        args.chunk_size = 15
        print("🎯 Using high quality preset")
    elif args.gpu:
        args.model = 'medium'
        args.device = 'auto'
        args.chunk_size = 25
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

    # Validate custom translation arguments up front. Any unknown service
    # name is treated as a custom OpenAI-compatible endpoint by the
    # translator, so it needs url+key regardless of the processing mode.
    known_services = set(UniversalTranslator.list_supported_models()) | {"custom"}
    if (args.translation_model not in known_services
            and (not args.translation_url or not args.translation_api_key)):
        print("Error: --translation-url and --translation-api-key are required when using custom model")
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
                language=args.language,
                target_language=args.target_language,
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

    if args.translate_only:
        # Translate-only mode: no transcription; the existing SRT is the source
        _run_translate_only(args, output_dir, keep_video_files)
        return

    # Single-file mode: delegate the whole flow (video-to-audio conversion,
    # transcription, translation, format conversion, video/MP3 exports) to
    # the shared one-shot pipeline. flat_output keeps the CLI's original
    # output layout (files written directly into --output-dir).
    processor = BatchProcessor(
        model_size=args.model,
        device=args.device,
        language=args.language,
        target_language=args.target_language,
        deepseek_model=args.translation_model,
        translation_api_key=args.translation_api_key,
        translation_url=args.translation_url,
        chunk_size=args.chunk_size,
        prompt_file=args.prompt_file,
        max_workers=args.max_workers,
        generate_video=args.generate_video,
        background_image=args.background_image,
        subtitle_position=args.subtitle_position,
        subtitle_fontsize=args.subtitle_fontsize,
        font_path=args.font_path,
        embed_mp3_metadata=args.embed_mp3_metadata,
        cover_image=args.cover_image,
        track_title=args.track_title,
        artist_name=args.artist_name,
        album_name=args.album_name,
        use_cache=not args.no_cache,
        initial_prompt=args.initial_prompt,
    )

    result = processor.process_single_file(
        args.audio_file,
        args.output_dir,
        convert_to=args.convert_to,
        keep_video_files=keep_video_files,
        flat_output=True,
    )

    if result['status'] == 'error':
        print(f"Error: {result['error']}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Processing completed successfully!")
    print("=" * 50)
    print(f"Original subtitles: {result['original_srt']}")
    if args.target_language != 'none':
        print(f"Translated subtitles ({args.target_language}): {result['translated_srt']}")
    if result.get('video_file'):
        print(f"Output video: {result['video_file']}")
    if result.get('mp3_file'):
        print(f"Output MP3: {result['mp3_file']}")

if __name__ == "__main__":
    main()