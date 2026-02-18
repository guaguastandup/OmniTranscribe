import whisper
import os
import re
import time
import threading
import torch
from datetime import timedelta
from typing import List, Optional, Callable

# Import cache module
try:
    from cache import get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

class AudioTranscriber:
    def __init__(self, model_size: str = "base", device: str = "auto", use_cache: bool = True):
        """
        Initialize the audio transcriber with Whisper model

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to use for inference ("auto", "cpu", "cuda", "mps")
            use_cache: Whether to use transcription cache (default: True)
        """
        self.model_size = model_size
        self.use_cache = use_cache and CACHE_AVAILABLE
        if self.use_cache:
            self.cache = get_cache()
        else:
            self.cache = None
        # Determine the device to use
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"GPU detected: {torch.cuda.get_device_name()}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("Apple Metal Performance Shaders (MPS) detected")
            else:
                device = "cpu"
                print("No GPU detected, using CPU")

        # Validate device choice
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("MPS not available, falling back to CPU")
            device = "cpu"

        self.device = device
        print(f"Loading Whisper model '{model_size}' on device: {device}")

        # Load model on specified device
        self.model = whisper.load_model(model_size, device=device)
        print(f"Loaded Whisper model: {model_size} on {device}")

    def transcribe_audio(self, audio_path: str, language: str = "ja", progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """
        Transcribe audio file to Japanese text

        Args:
            audio_path: Path to the audio file
            language: Language code (default: 'ja' for Japanese)
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            SRT formatted subtitle content
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing audio file: {audio_path}")
        print(f"Using language: {language}")

        # Check cache first
        if self.use_cache and self.cache:
            cached_result = self.cache.get(audio_path, self.model_size, language)
            if cached_result is not None:
                print("Using cached transcription result!")
                return cached_result
            else:
                print("No cached result found, proceeding with transcription...")

        # Get audio duration for progress estimation
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            print(f"Audio duration: {duration:.1f} seconds")
        except ImportError:
            # If librosa is not available, use a rough estimation based on file size
            file_size = os.path.getsize(audio_path)
            # Rough estimation: 1MB ≈ 1 minute for typical audio
            duration = file_size / (1024 * 1024) * 60
            print(f"Audio duration (estimated): {duration:.1f} seconds")

        # Progress display disabled - removing incorrect progress bar
        stop_event = threading.Event()
        # progress_thread = threading.Thread(
        #     target=self._show_progress,
        #     args=(duration, stop_event, progress_callback)
        # )
        # progress_thread.daemon = True
        # progress_thread.start()

        try:
            # Transcribe the audio with enhanced Japanese settings
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                word_timestamps=True,
                # Additional parameters to improve Japanese transcription
                temperature=0.0,  # Lower temperature for more consistent results
                best_of=5,        # Try more candidates for better accuracy
                beam_size=5,      # Use beam search for better results
                patience=1.0,     # Beam search patience
                # Force Japanese language detection
                # language_detection_threshold=0.1,
                # Disable translation to ensure we get original Japanese
                condition_on_previous_text=True,
                # Suppress tokens that might lead to English
                suppress_tokens="-1",
                # Use Japanese-specific prompt if needed
                initial_prompt=""
            )

            # Stop progress thread (disabled)
            stop_event.set()
            # progress_thread.join(timeout=1)

            # Convert to SRT format
            srt_content = self._convert_to_srt(result["segments"])

            # Save to cache
            if self.use_cache and self.cache:
                self.cache.set(audio_path, self.model_size, srt_content, language)

            # # Validate that the transcription is in Japanese
            # if self._contains_english_text(srt_content):
            #     print("Warning: Detected English text in transcription. Re-transcribing with stricter Japanese settings...")
            #     # Retry with even stricter Japanese settings
            #     return self._transcribe_strict_japanese(audio_path, result["segments"])

            return srt_content

        except Exception as e:
            # Stop progress thread on error (disabled)
            stop_event.set()
            # progress_thread.join(timeout=1)
            raise e

    def _show_progress(self, duration: float, stop_event: threading.Event, callback: Optional[Callable[[int, int], None]]):
        """
        Show progress during transcription using time-based estimation

        Args:
            duration: Total audio duration in seconds
            stop_event: Event to stop the progress display
            callback: Progress callback function
        """
        try:
            from tqdm import tqdm
            import sys

            # Create a progress bar
            pbar = tqdm(
                total=100,
                desc="Transcribing",
                unit="%",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                file=sys.stdout,
                ncols=80  # Limit width for better display
            )

            # Estimate transcription time based on model and audio duration
            # Whisper models typically process faster than real-time
            transcription_factor = 0.2  # Conservative estimate: 20% of real-time
            estimated_time = duration * transcription_factor

            start_time = time.time()
            last_progress = 0

            while not stop_event.is_set():
                elapsed = time.time() - start_time
                if estimated_time > 0:
                    progress = min(100, int((elapsed / estimated_time) * 100))
                else:
                    # Fallback: very slow linear progress
                    progress = min(100, int((elapsed / (duration * 0.5)) * 100))

                # Only update if progress increased significantly
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress

                # Complete if we reach 100%
                if progress >= 100:
                    break

                time.sleep(0.3)  # Update every 0.3 seconds

            # Ensure progress bar completes when transcription finishes
            if last_progress < 100:
                pbar.update(100 - last_progress)

            pbar.close()

        except ImportError:
            # Fallback if tqdm is not available
            print("Transcribing... (this may take a while)")
            start_time = time.time()
            estimated_time = duration * 0.2

            while not stop_event.is_set():
                elapsed = time.time() - start_time
                if estimated_time > 0:
                    progress = min(100, int((elapsed / estimated_time) * 100))
                else:
                    progress = min(100, int((elapsed / (duration * 0.5)) * 100))

                print(f"\rTranscription progress: {progress:3d}%", end="", flush=True)

                if progress >= 100:
                    break

                time.sleep(1)  # Update every second

            print()  # New line after completion

    def _convert_to_srt(self, segments: List[dict]) -> str:
        """
        Convert Whisper segments to SRT format

        Args:
            segments: List of segments from Whisper

        Returns:
            SRT formatted string
        """
        srt_lines = []

        for i, segment in enumerate(segments, 1):
            # Convert seconds to time strings
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])

            # Add subtitle entry
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment['text'].strip())
            srt_lines.append("")  # Empty line between entries

        # Join all lines with newlines
        return "\n".join(srt_lines)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm)

        Args:
            seconds: Time in seconds

        Returns:
            SRT time format string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def save_srt(self, content: str, output_path: str) -> None:
        """
        Save SRT content to file

        Args:
            content: SRT content to save
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved original subtitles to: {output_path}")

    def _contains_english_text(self, text: str) -> bool:
        """
        Check if the transcribed text contains significant English content

        Args:
            text: Transcribed text

        Returns:
            True if significant English text is detected
        """
        # Remove SRT formatting
        clean_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', text)
        clean_text = re.sub(r'\n+', ' ', clean_text).strip()

        if not clean_text:
            return False

        # Count English words (basic Latin characters)
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', clean_text))
        total_words = len(re.findall(r'\b[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+\b', clean_text))

        # If more than 20% of words are English, consider it problematic
        if total_words > 0 and (english_words / total_words) > 0.2:
            print(f"Detected {english_words} English words out of {total_words} total words")
            return True

        return False

    def _transcribe_strict_japanese(self, audio_path: str, previous_segments: List[dict]) -> str:
        """
        Re-transcribe with very strict Japanese settings

        Args:
            audio_path: Path to the audio file
            previous_segments: Previous transcription segments for context

        Returns:
            SRT formatted string
        """
        print("Using strict Japanese transcription mode...")

        # Create a Japanese-only prompt
        japanese_prompt = "これは日本語の音声です。ひらがな、カタカナ、漢字のみを使用して文字起こししてください。英語は使用しないでください。"

        # If we have previous segments, add them as context
        if previous_segments:
            previous_text = " ".join([seg['text'].strip() for seg in previous_segments[-3:]])  # Last 3 segments
            japanese_prompt += f"\n前の文脈：{previous_text}"

        # Transcribe with very strict settings
        result = self.model.transcribe(
            audio_path,
            language="ja",
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,  # No randomness
            best_of=1,        # Single best result
            beam_size=1,      # No beam search
            condition_on_previous_text=False,  # Don't use previous text to avoid English influence
            suppress_tokens="-1",  # Suppress all non-Japanese tokens
            initial_prompt=japanese_prompt,
            # Additional strict parameters
            no_speech_threshold=0.1,
            logprob_threshold=-1.0
        )

        srt_content = self._convert_to_srt(result["segments"])

        # Final validation
        if self._contains_english_text(srt_content):
            print("Warning: Still detecting English text. Using fallback method...")
            return self._fallback_japanese_transcription(result["segments"])

        return srt_content

    def _fallback_japanese_transcription(self, segments: List[dict]) -> str:
        """
        Fallback method to force Japanese transcription

        Args:
            segments: Whisper segments

        Returns:
            SRT formatted string
        """
        print("Applying fallback Japanese text processing...")

        srt_lines = []

        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])

            # Process text to remove/convert English
            text = segment['text'].strip()

            # Remove common English words and patterns
            text = re.sub(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', '', text, flags=re.IGNORECASE)

            # Convert any remaining English characters to Japanese equivalents if possible
            text = self._convert_to_japanese_equiv(text)

            # Clean up extra spaces
            text = re.sub(r'\s+', ' ', text).strip()

            if text:  # Only add if there's content left
                srt_lines.append(str(i))
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")  # Empty line between entries

        return "\n".join(srt_lines)

    def _convert_to_japanese_equiv(self, text: str) -> str:
        """
        Convert some English words/characters to rough Japanese equivalents

        Args:
            text: Input text

        Returns:
            Text with some conversions applied
        """
        # Basic character replacements (this is a simple fallback)
        replacements = {
            'hello': 'こんにちは',
            'thank you': 'ありがとう',
            'yes': 'はい',
            'no': 'いいえ',
            'sorry': 'すみません',
            'please': 'お願いします',
            'good': '良い',
            'bad': '悪い',
            'love': '愛',
            'like': '好き',
            'want': '欲しい',
            'can': 'できる',
            'cannot': 'できない',
        }

        for eng, jpn in replacements.items():
            text = re.sub(r'\b' + re.escape(eng) + r'\b', jpn, text, flags=re.IGNORECASE)

        return text