import asyncio
import json
from dataclasses import replace

from app.answerer import (
    INSTRUCTIONS,
    PACK_INSTRUCTIONS,
    SUMMARY_INSTRUCTIONS,
    Answerer,
    Context,
    build_input,
    improve_request,
    supports_cache_options,
)
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


def test_pack_and_summary_use_the_brief_model(settings):
    client = FakeClient()
    answerer = Answerer(client, replace(settings, brief_model="gpt-5-mini", brief_effort="low"))
    usage = UsageTracker()
    assert asyncio.run(answerer.make_pack("MATERIAL", usage=usage)) == "- Participants: Dana"
    asyncio.run(answerer.summarize_meeting("MEETING", usage=usage))
    pack, summary = client.responses.calls
    assert pack["instructions"] == PACK_INSTRUCTIONS and pack["input"] == "MATERIAL"
    assert summary["instructions"] == SUMMARY_INSTRUCTIONS and summary["input"] == "MEETING"
    assert pack["model"] == summary["model"] == "gpt-5-mini" and pack["store"] is False
    assert "## Question bank" in PACK_INSTRUCTIONS and "## 中文要点" in SUMMARY_INSTRUCTIONS
    assert "question bank" in INSTRUCTIONS and "story bank" in INSTRUCTIONS


def test_detail_uses_the_stronger_model_with_short_thinking(settings):
    client = FakeClient()
    answerer = Answerer(client, settings)

    async def run():
        stream = answerer.stream(Context(background="BG"), "Q", cache_key="k", usage=UsageTracker(), deep=True)
        return [e.text async for e in stream]

    assert "".join(asyncio.run(run())) == "I led the migration."
    [call] = client.responses.calls
    assert call["model"] == "gpt-6-sol" and call["reasoning"] == {"effort": "low"}
    assert call["max_output_tokens"] == 3000
    assert client.responses.streams[0].closed


def test_detail_falls_back_to_the_fast_model_when_thinking_runs_long(settings):
    client = FakeClient(delay=0.3)  # the thinking run shows no words for 0.3 s
    answerer = Answerer(client, replace(settings, detail_think_s=0.05))
    create = client.responses.create

    async def create_then_speed_up(**kwargs):
        stream = await create(**kwargs)
        client.responses.delay = 0.0  # the fallback answers at once
        return stream

    client.responses.create = create_then_speed_up

    async def run():
        stream = answerer.stream(Context(), "Q", cache_key="k", usage=UsageTracker(), deep=True)
        return [(e.kind, e.text) async for e in stream]

    events = asyncio.run(run())
    assert events[0] == ("status", "quick")
    assert "".join(text for kind, text in events if kind == "delta") == "I led the migration."
    slow, fast = client.responses.calls
    assert slow["model"] == "gpt-6-sol" and fast["model"] == "gpt-6-luna"
    assert fast["reasoning"] == {"effort": "none"} and fast["max_output_tokens"] == 3000
    assert all(s.closed for s in client.responses.streams)  # the dropped thinking run stops costing tokens
