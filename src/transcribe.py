import whisper
import os
import threading
import torch
from typing import Any, Dict, List, Optional, Callable, Tuple

# Import cache module
try:
    from cache import get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Loaded Whisper models, keyed by (model_size, device). Models are expensive to
# load (tens of seconds for medium/large), but safe to share for inference, so
# resuming a project reuses the model instead of reloading it.
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()

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

        # Reuse a previously loaded model so resuming a project doesn't pay
        # the model-load cost again (whisper models are safe to share).
        model_key = (model_size, device)
        with _MODEL_CACHE_LOCK:
            if model_key not in _MODEL_CACHE:
                print(f"Loading Whisper model '{model_size}' on device: {device}")
                _MODEL_CACHE[model_key] = whisper.load_model(model_size, device=device)
                print(f"Loaded Whisper model: {model_size} on {device}")
            else:
                print(f"Reusing loaded Whisper model '{model_size}' on device: {device}")
        self.model = _MODEL_CACHE[model_key]

    def transcribe_audio(self, audio_path: str, language: str = "auto", progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to the audio file
            language: Language code (default: 'auto' for auto-detection)
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            SRT formatted subtitle content
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Whisper auto-detects the language only when language is None;
        # it raises "Unsupported language" for "auto".
        whisper_language = None if language in ("auto", "") else language

        print(f"Transcribing audio file: {audio_path}")
        if whisper_language is None:
            print("Using language: auto-detection")
        else:
            print(f"Using language: {whisper_language}")

        # Check cache first
        if self.use_cache and self.cache:
            cached_result = self.cache.get(audio_path, self.model_size, language)
            if cached_result is not None:
                print("Using cached transcription result!")
                return cached_result
            else:
                print("No cached result found, proceeding with transcription...")

        # Get audio duration for progress estimation (informational only)
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

        result = self.model.transcribe(
            audio_path,
            language=whisper_language,
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,  # Lower temperature for more consistent results
            beam_size=5,      # Use beam search for better results
            patience=1.0,     # Beam search patience
            condition_on_previous_text=True,
            suppress_tokens="-1",
        )

        # Convert to SRT format
        srt_content = self._convert_to_srt(result["segments"])

        # Save to cache
        if self.use_cache and self.cache:
            self.cache.set(audio_path, self.model_size, srt_content, language)

        return srt_content


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