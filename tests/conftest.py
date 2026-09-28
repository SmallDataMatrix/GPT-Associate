"""Test doubles for the OpenAI client and the transcriber, so tests run offline and fast."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.answerer import Answerer
from app.config import Settings
from app.meeting import MeetingSession
from app.store import Store


def fake_usage(input_tokens=1200, cached=1000, output=40):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


class FakeStream:
    def __init__(self, chunks, delay):
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield SimpleNamespace(type="response.output_text.delta", delta=chunk)
        yield SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=fake_usage()))

    async def close(self):
        self.closed = True


class FakeResponses:
    def __init__(self, chunks=("I led ", "the migration."), delay=0.0, text="- Participants: Dana"):
        self.chunks = list(chunks)
        self.delay = delay
        self.text = text
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            stream = FakeStream(self.chunks, self.delay)
            self.streams.append(stream)
            return stream
        return SimpleNamespace(output_text=self.text, usage=fake_usage(output=80))


class FakeClient:
    def __init__(self, **kwargs):
        self.responses = FakeResponses(**kwargs)


class FakeTranscriber:
    instances: list["FakeTranscriber"] = []

    def __init__(self, source, settings, handler, terms=None):
        self.source, self.settings, self.handler, self.terms = source, settings, handler, terms or []
        self.chunks: list[bytes] = []
        self.started = self.stopped = False
        FakeTranscriber.instances.append(self)

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def push(self, pcm):
        self.chunks.append(pcm)

    async def set_terms(self, terms):
        self.terms = terms


def text_of(item: dict) -> str:
    """Text of a Responses API input item whose content is a list of input_text blocks."""
    return "".join(block["text"] for block in item["content"])


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), data_dir=tmp_path / "data", access_code="123456", openai_api_key="test")


def make_session(settings, **client_kwargs) -> tuple[MeetingSession, FakeClient]:
    client = FakeClient(**client_kwargs)
    session = MeetingSession(settings, Store(settings.data_dir), Answerer(client, settings), FakeTranscriber)
    return session, client


async def settle(session: MeetingSession, timeout: float = 5.0) -> None:
    """Wait until every answer and background task has finished."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while session._tasks or session._background:
        if loop.time() > deadline:
            raise TimeoutError("tasks did not finish")
        await asyncio.sleep(0.01)
