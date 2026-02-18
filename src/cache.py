#!/usr/bin/env python3
"""
Cache management for OmniTranscribe.

This module provides caching functionality for transcription results
based on file hash and Whisper model size to avoid re-transcribing
the same audio files.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class TranscriptionCache:
    """Cache for transcription results based on file hash and model."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the transcription cache.

        Args:
            cache_dir: Directory to store cache files (default: ~/.cache/omnitranscribe)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "omnitranscribe"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache index: {e}")
                return {}
        return {}

    def _save_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save cache index: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_cache_key(self, file_path: str, model_size: str, language: str = "auto") -> str:
        """
        Generate a unique cache key for a transcription.

        Args:
            file_path: Path to the audio file
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Source language code

        Returns:
            Cache key string
        """
        file_hash = self._get_file_hash(file_path)
        return f"{file_hash}:{model_size}:{language}"

    def get(self, file_path: str, model_size: str, language: str = "auto") -> Optional[str]:
        """
        Get cached transcription if available.

        Args:
            file_path: Path to the audio file
            model_size: Whisper model size
            language: Source language code

        Returns:
            Cached SRT content if found, None otherwise
        """
        cache_key = self.get_cache_key(file_path, model_size, language)

        if cache_key in self.index:
            entry = self.index[cache_key]
            cached_file = Path(entry["cached_file"])

            if cached_file.exists():
                print(f"✓ Using cached transcription from {entry['timestamp']}")
                try:
                    with open(cached_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f"Warning: Failed to read cached file: {e}")
            else:
                # Remove stale entry
                del self.index[cache_key]
                self._save_index()

        return None

    def set(self, file_path: str, model_size: str, srt_content: str, language: str = "auto"):
        """
        Cache a transcription result.

        Args:
            file_path: Path to the audio file
            model_size: Whisper model size
            srt_content: SRT content to cache
            language: Source language code
        """
        cache_key = self.get_cache_key(file_path, model_size, language)

        # Generate cache filename
        file_hash = self._get_file_hash(file_path)
        cache_filename = f"{file_hash[:16]}_{model_size}_{language}.srt"
        cache_file_path = self.cache_dir / cache_filename

        # Save SRT content
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            # Update index
            self.index[cache_key] = {
                "cached_file": str(cache_file_path),
                "timestamp": datetime.now().isoformat(),
                "model_size": model_size,
                "language": language,
                "original_file": str(file_path),
                "original_filename": Path(file_path).name
            }
            self._save_index()

            print(f"✓ Cached transcription to: {cache_file_path}")

        except Exception as e:
            print(f"Warning: Failed to cache transcription: {e}")

    def clear(self):
        """Clear all cached transcriptions."""
        # Remove all cache files except index
        for cache_key, entry in self.index.items():
            cache_file = Path(entry["cached_file"])
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete cache file {cache_file}: {e}")

        # Clear index
        self.index = {}
        self._save_index()
        print("✓ Cache cleared")

    def list_cached(self) -> list:
        """
        List all cached transcriptions.

        Returns:
            List of cache entry dictionaries
        """
        return [
            {
                "key": key,
                "timestamp": entry["timestamp"],
                "model": entry["model_size"],
                "language": entry["language"],
                "filename": entry["original_filename"]
            }
            for key, entry in self.index.items()
        ]

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self.index)
        total_size = 0

        for entry in self.index.values():
            cache_file = Path(entry["cached_file"])
            if cache_file.exists():
                total_size += cache_file.stat().st_size

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance
_cache = None


def get_cache() -> TranscriptionCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = TranscriptionCache()
    return _cache
