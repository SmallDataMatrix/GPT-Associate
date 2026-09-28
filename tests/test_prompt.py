import asyncio
import json
from dataclasses import replace

from app.answerer import INSTRUCTIONS, Answerer, Context, build_input, improve_request, supports_cache_options
from app.usage import UsageTracker

from .conftest import FakeClient, text_of


def test_static_prefix_is_identical_across_calls():
    """Prompt caching only works if everything before the dynamic part is byte-identical."""
    first = build_input(Context(background="BG", notes="n1", transcript="Them: hi"), "Q1", breakpoints=True)
    second = build_input(
        Context(background="BG", notes="n1", transcript="Them: hi\nMe: hello", excerpts=["x"]), "Q2", breakpoints=True
    )
    assert json.dumps(first[:2]) == json.dumps(second[:2])
    assert first[2:] != second[2:]


def test_order_is_background_notes_conversation_request():
    items = build_input(Context(background="BG", notes="NOTES", transcript="Them: hi", excerpts=["EXCERPT"]), "REQ")
    assert [i["role"] for i in items] == ["developer", "developer", "developer", "user"]
    assert "BG" in text_of(items[0]) and text_of(items[1]) == "NOTES"
    conversation = text_of(items[2])
    assert conversation.index("Them: hi") < conversation.index("EXCERPT")
    assert text_of(items[3]) == "REQ"


def test_breakpoints_mark_background_and_notes_only():
    items = build_input(Context(background="BG", notes="NOTES", transcript="t"), "REQ", breakpoints=True)
    marked = [bool(i["content"][0].get("prompt_cache_breakpoint")) for i in items]
    assert marked == [True, True, False, False]
    assert not any("prompt_cache_breakpoint" in i["content"][0] for i in build_input(Context(), "REQ"))


def test_cache_option_support_by_model():
    assert supports_cache_options("gpt-6-luna") and supports_cache_options("gpt-5.6-terra")
    assert not supports_cache_options("gpt-5-mini") and not supports_cache_options("gpt-4.1-mini")


def test_improve_request_uses_my_reply_or_falls_back():
    with_reply = improve_request("Why us?", "I admire your mission.", "Because of the pay.")
    assert "Because of the pay." in with_reply and "Missed / weak:" in with_reply and "Add now:" in with_reply
    without = improve_request("Why us?", "I admire your mission.", "")
    assert "not captured" in without and "I admire your mission." in without


def test_stream_sends_cache_friendly_request(settings):
    client = FakeClient()
    answerer = Answerer(client, settings)
    usage = UsageTracker()

    async def run():
        return [e.text async for e in answerer.stream(Context(background="BG"), "Q", cache_key="ga-x", usage=usage)]

    assert "".join(asyncio.run(run())) == "I led the migration."
    call = client.responses.calls[0]
    assert call["instructions"] == INSTRUCTIONS
    assert call["prompt_cache_key"] == "ga-x"
    assert call["reasoning"] == {"effort": "none"}
    assert call["store"] is False and call["stream"] is True
    assert "tools" not in call
    assert call["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert client.responses.streams[0].closed
    assert usage.public()["cached_pct"] == 83


def test_prewarm_uses_cache_prewarm_on_new_models_only(settings):
    async def run(model):
        client = FakeClient()
        answerer = Answerer(client, replace(settings, answer_model=model))
        await answerer.prewarm(Context(background="BG"), cache_key="k", usage=UsageTracker())
        return client.responses.calls[0]

    assert asyncio.run(run("gpt-6-luna"))["prompt_cache_options"] == {"prewarm": True}
    old = asyncio.run(run("gpt-4.1-mini"))
    assert "prompt_cache_options" not in old
    assert "prompt_cache_breakpoint" not in old["input"][0]["content"][0]


def test_reasoning_is_omitted_when_effort_is_blank(settings):
    client = FakeClient()
    answerer = Answerer(client, replace(settings, notes_model="gpt-4.1-mini", notes_effort=""))
    asyncio.run(answerer.update_notes("", "Them: hi", usage=UsageTracker()))
    assert "reasoning" not in client.responses.calls[0]
