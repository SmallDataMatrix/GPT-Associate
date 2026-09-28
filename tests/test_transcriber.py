import asyncio
import base64
import json
from dataclasses import replace

import websockets

from app.transcriber import Transcriber, realtime_url, session_config, transcription_config


class Recorder:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *args: self.events.append((name, *args))
        raise AttributeError(name)


def test_realtime_url():
    assert realtime_url() == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert realtime_url("http://127.0.0.1:9/v1/") == "ws://127.0.0.1:9/v1/realtime?intent=transcription"


def test_session_config_shape(settings):
    config = session_config("me", settings, ["Kafka"])
    audio = config["session"]["audio"]["input"]
    assert config["type"] == "session.update" and config["session"]["type"] == "transcription"
    assert audio["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio["transcription"]["model"] == "gpt-4o-mini-transcribe"
    assert audio["transcription"]["language"] == "en" and "Kafka" in audio["transcription"]["prompt"]
    assert audio["turn_detection"]["type"] == "server_vad"
    assert audio["turn_detection"]["silence_duration_ms"] == 500
    assert audio["noise_reduction"] == {"type": "near_field"}
    assert session_config("them", settings)["session"]["audio"]["input"]["noise_reduction"] is None


def test_gpt_transcribe_gets_keywords():
    assert transcription_config("gpt-transcribe", ["Kafka"]) == {
        "model": "gpt-transcribe", "languages": ["en"], "keywords": ["Kafka"],
    }


def test_relays_audio_dispatches_events_and_reconnects(settings):
    async def run():
        received = []
        connections = 0

        async def fake_openai(ws):
            nonlocal connections
            connections += 1
            received.append(("headers", ws.request.headers.get("Authorization"), ws.request.path))
            received.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({"type": "session.updated"}))
            received.append(json.loads(await ws.recv()))
            if connections == 1:
                for event in (
                    {"type": "input_audio_buffer.speech_started", "item_id": "a"},
                    {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": "Hel"},
                    {"type": "input_audio_buffer.speech_stopped", "item_id": "a"},
                    {"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "Hello."},
                ):
                    await ws.send(json.dumps(event))
                await ws.close()  # like a Realtime session reaching its time limit
            else:
                async for _ in ws:  # keep consuming audio like the real service
                    pass

        async with websockets.serve(fake_openai, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            handler = Recorder()
            transcriber = Transcriber("them", replace(settings, openai_base_url=f"http://127.0.0.1:{port}/v1"), handler, ["Acme"])
            transcriber.start()
            transcriber.push(b"\x01\x00" * 2400)
            for _ in range(200):
                if connections == 2 and len(received) >= 6:
                    break
                if connections == 1 and len(received) >= 3:
                    transcriber.push(b"\x02\x00" * 2400)  # audio for the reconnected session
                await asyncio.sleep(0.02)
            await transcriber.stop()
            return received, handler.events

    received, events = asyncio.run(run())
    assert received[0] == ("headers", "Bearer test", "/v1/realtime?intent=transcription")
    assert received[1]["type"] == "session.update"
    assert received[2]["type"] == "input_audio_buffer.append"
    assert base64.b64decode(received[2]["audio"]) == b"\x01\x00" * 2400
    assert received[4]["type"] == "session.update"  # config re-sent after reconnecting
    names = [e[0] for e in events]
    assert ("on_partial", "them", "a", "Hel") in events
    assert ("on_final", "them", "a", "Hello.") in events
    assert names.index("on_speech_started") < names.index("on_speech_stopped") < names.index("on_final")
    assert ("on_stt_status", "them", "reconnecting") in events
    assert ("on_stt_status", "them", "listening") in events
