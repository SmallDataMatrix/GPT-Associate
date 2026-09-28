"""Settings loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

# USD per 1M tokens: (input, cached input, output). Models not listed still count tokens, at no cost.
LLM_PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}
# USD per minute of audio.
STT_PRICES = {
    "gpt-4o-mini-transcribe": 0.003,
    "gpt-transcribe": 0.0045,
    "gpt-4o-transcribe": 0.006,
    "whisper-1": 0.006,
}
WEB_SEARCH_PRICE = 0.01  # per call; search content tokens are billed as input tokens


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    return default if not value else value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_base_url: str = ""
    answer_model: str = "gpt-6-luna"
    answer_effort: str = "none"
    answer_max_tokens: int = 400
    notes_model: str = "gpt-6-luna"
    notes_effort: str = "none"
    brief_model: str = "gpt-6-luna"
    brief_effort: str = "low"
    transcribe_model: str = "gpt-4o-mini-transcribe"
    vad_silence_ms: int = 500
    vad_threshold: float = 0.5
    full_text_token_limit: int = 15000
    lan_access: bool = True
    access_code: str = "000000"
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = ROOT / "data"


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    lan = _bool("LAN_ACCESS", True)
    return Settings(
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        answer_model=_env("ANSWER_MODEL", "gpt-6-luna"),
        answer_effort=_env("ANSWER_REASONING_EFFORT", "none"),
        answer_max_tokens=int(_env("ANSWER_MAX_TOKENS", "400")),
        notes_model=_env("NOTES_MODEL", "gpt-6-luna"),
        notes_effort=_env("NOTES_REASONING_EFFORT", "none"),
        brief_model=_env("BRIEF_MODEL", "gpt-6-luna"),
        brief_effort=_env("BRIEF_REASONING_EFFORT", "low"),
        transcribe_model=_env("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
        vad_silence_ms=int(_env("VAD_SILENCE_MS", "500")),
        vad_threshold=float(_env("VAD_THRESHOLD", "0.5")),
        full_text_token_limit=int(_env("FULL_TEXT_TOKEN_LIMIT", "15000")),
        lan_access=lan,
        access_code=_env("ACCESS_CODE") or f"{secrets.randbelow(10**6):06d}",
        host="0.0.0.0" if lan else "127.0.0.1",
        port=int(_env("PORT", "8000")),
        data_dir=Path(_env("DATA_DIR") or ROOT / "data"),
    )
