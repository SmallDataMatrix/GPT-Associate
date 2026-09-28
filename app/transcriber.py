"""Relays one audio source to an OpenAI Realtime transcription session and reports transcript events.

The browser sends 24 kHz mono PCM16 over /ws/audio; this class forwards it to
wss://api.openai.com/v1/realtime?intent=transcription with server-side VAD, and reconnects on its own
when the session ends or the network drops (Realtime sessions have a maximum length).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Protocol

import websockets

from .config import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
QUEUE_CHUNKS = 300  # about 30 s of 100 ms chunks buffered while reconnecting
# Tab/system audio is already clean; microphones benefit from noise reduction.
NOISE_REDUCTION = {"me": "near_field", "room": "far_field"}
KEYWORD_MODELS = ("gpt-transcribe", "gpt-live-transcribe")


class TranscriptHandler(Protocol):
    def on_speech_started(self, source: str, item_id: str) -> None: ...
    def on_speech_stopped(self, source: str, item_id: str) -> None: ...
    def on_partial(self, source: str, item_id: str, delta: str) -> None: ...
    def on_final(self, source: str, item_id: str, text: str) -> None: ...
    def on_stt_status(self, source: str, status: str) -> None: ...
    def on_stt_error(self, source: str, message: str) -> None: ...
    def on_audio_seconds(self, source: str, seconds: float) -> None: ...


def realtime_url(base_url: str = "") -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/realtime?intent=transcription"


def transcription_config(model: str, terms: list[str]) -> dict:
    """gpt-transcribe models take a keyword list; the gpt-4o transcribe models take a free-text prompt."""
    if model.startswith(KEYWORD_MODELS):
        config: dict = {"model": model, "languages": ["en"]}
        if terms:
            config["keywords"] = terms
        return config
    prompt = "English conversation in a professional meeting or job interview."
    if terms:
        prompt += f" Names and terms that may come up: {', '.join(terms)}."
    return {"model": model, "language": "en", "prompt": prompt}


def session_config(source: str, settings: Settings, terms: list[str] | None = None) -> dict:
    transcription = transcription_config(settings.transcribe_model, terms or [])
    noise = NOISE_REDUCTION.get(source)
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": transcription,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": settings.vad_threshold,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": settings.vad_silence_ms,
                    },
                    "noise_reduction": {"type": noise} if noise else None,
                }
            },
        },
    }


class Transcriber:
    def __init__(self, source: str, settings: Settings, handler: TranscriptHandler, terms: list[str] | None = None) -> None:
        self.source = source
        self.settings = settings
        self.handler = handler
        self.terms = terms or []
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=QUEUE_CHUNKS)
        self._task: asyncio.Task | None = None
        self._ws = None
        self._closing = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"transcriber-{self.source}")

    async def stop(self) -> None:
        self._closing = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def push(self, pcm: bytes) -> None:
        if self._queue.full():
            self._queue.get_nowait()  # drop the oldest audio rather than fall further behind
        self._queue.put_nowait(pcm)

    async def set_terms(self, terms: list[str]) -> None:
        self.terms = terms
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps(session_config(self.source, self.settings, terms)))
            except Exception:  # the reconnect loop re-sends the config anyway
                pass

    async def _run(self) -> None:
        delay = 0.5
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        url = realtime_url(self.settings.openai_base_url)
        while not self._closing:
            try:
                self.handler.on_stt_status(self.source, "connecting")
                async with websockets.connect(
                    url, additional_headers=headers, max_size=None, open_timeout=10, close_timeout=2
                ) as ws:
                    self._ws = ws
                    await ws.send(json.dumps(session_config(self.source, self.settings, self.terms)))
                    sender = asyncio.create_task(self._send_audio(ws))
                    try:
                        async for message in ws:
                            self._handle(json.loads(message))
                            delay = 0.5
                    finally:
                        sender.cancel()
                        self._ws = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("%s transcription connection failed: %s", self.source, exc)
                self.handler.on_stt_error(self.source, f"Transcription connection failed: {exc}")
            if self._closing:
                break
            self.handler.on_stt_status(self.source, "reconnecting")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

    async def _send_audio(self, ws) -> None:
        while True:
            chunk = await self._queue.get()
            self.handler.on_audio_seconds(self.source, len(chunk) / BYTES_PER_SECOND)
            event = {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}
            await ws.send(json.dumps(event))

    def _handle(self, event: dict) -> None:
        kind = event.get("type", "")
        item_id = event.get("item_id", "")
        if kind == "conversation.item.input_audio_transcription.delta":
            self.handler.on_partial(self.source, item_id, event.get("delta") or "")
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.handler.on_final(self.source, item_id, event.get("transcript") or "")
        elif kind == "input_audio_buffer.speech_started":
            self.handler.on_speech_started(self.source, item_id)
        elif kind == "input_audio_buffer.speech_stopped":
            self.handler.on_speech_stopped(self.source, item_id)
        elif kind == "conversation.item.input_audio_transcription.failed":
            log.warning("%s transcription failed: %s", self.source, event.get("error"))
            self.handler.on_final(self.source, item_id, "")
            # Show it on the page too: otherwise a quota or model problem looks like "listening" with no text.
            message = (event.get("error") or {}).get("message") or "Transcription failed"
            self.handler.on_stt_error(self.source, message)
        elif kind in ("session.created", "session.updated", "transcription_session.created",
                      "transcription_session.updated"):
            self.handler.on_stt_status(self.source, "listening")
        elif kind == "error":
            message = (event.get("error") or {}).get("message") or "Unknown transcription error"
            log.error("%s transcription error: %s", self.source, message)
            self.handler.on_stt_error(self.source, message)
