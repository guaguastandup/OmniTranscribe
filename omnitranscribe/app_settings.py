from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

try:
    import keyring
except ImportError:  # pragma: no cover
    keyring = None

try:
    from platformdirs import user_config_dir
except ImportError:  # pragma: no cover
    user_config_dir = None


SERVICE_NAME = "OmniTranscribe"
API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "QWEN_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
}


def _config_dir() -> Path:
    if user_config_dir:
        return Path(user_config_dir("OmniTranscribe", "OmniTranscribe"))
    return Path.home() / ".omnitranscribe"


class AppSettings:
    """User-facing settings with OS keyring support and a local fallback."""

    def __init__(self):
        self.config_dir = _config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_secret_file = self.config_dir / "secrets.json"

    def _read_fallback(self) -> Dict[str, str]:
        if not self.fallback_secret_file.exists():
            return {}
        try:
            data = json.loads(self.fallback_secret_file.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items() if v}
        except Exception:
            return {}

    def _write_fallback(self, data: Dict[str, str]) -> None:
        self.fallback_secret_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            os.chmod(self.fallback_secret_file, 0o600)
        except OSError:
            pass

    def get_secret(self, provider: str) -> str:
        env_name = API_KEY_ENV.get(provider)
        if not env_name:
            return ""
        if os.getenv(env_name):
            return os.getenv(env_name, "")
        if keyring is not None:
            try:
                value = keyring.get_password(SERVICE_NAME, env_name)
                if value:
                    return value
            except Exception:
                pass
        return self._read_fallback().get(env_name, "")

    def set_secret(self, provider: str, value: Optional[str]) -> None:
        env_name = API_KEY_ENV.get(provider)
        if not env_name:
            raise KeyError(f"Unknown translation provider: {provider}")
        value = (value or "").strip()
        stored = False
        if keyring is not None:
            try:
                if value:
                    keyring.set_password(SERVICE_NAME, env_name, value)
                else:
                    try:
                        keyring.delete_password(SERVICE_NAME, env_name)
                    except Exception:
                        pass
                stored = True
            except Exception:
                stored = False
        if not stored:
            fallback = self._read_fallback()
            if value:
                fallback[env_name] = value
            else:
                fallback.pop(env_name, None)
            self._write_fallback(fallback)
        if value:
            os.environ[env_name] = value
        else:
            os.environ.pop(env_name, None)

    def apply_to_environment(self) -> None:
        for provider, env_name in API_KEY_ENV.items():
            if not os.getenv(env_name):
                value = self.get_secret(provider)
                if value:
                    os.environ[env_name] = value

    def configured_providers(self) -> Dict[str, bool]:
        return {provider: bool(self.get_secret(provider)) for provider in API_KEY_ENV}
