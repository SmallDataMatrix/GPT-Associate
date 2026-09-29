import asyncio
import time

from .conftest import make_session, settle, text_of


def say(session, source, item_id, text, start, stop):
    """Feeds one finished transcript segment as the Transcriber would (VAD times, then the transcript)."""
    session.item_times[item_id] = {"start": start, "stop": stop}
    session.on_final(source, item_id, text)


def test_question_from_them_is_answered_automatically(settings):
    async def run():
        session, client = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Can you tell me about your last project?", now - 3, now - 0.1)
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    [card] = session.cards
    assert card.status == "done" and card.text == "I led the migration."
    assert card.question == "Can you tell me about your last project?"
    assert card.first_ms is not None and card.done_ms >= card.first_ms
    assert "They just said" in text_of(client.responses.calls[0]["input"][-1])


def test_statements_do_not_trigger_answers(settings):
    async def run():
        session, _ = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "We are a team of twelve based in Berlin.", now - 3, now - 0.1)
        await settle(session)
        return session

    session = asyncio.run(run())
    assert session.cards == [] and len(session.transcript.segments) == 1


def test_continuation_restarts_the_same_card_with_the_full_question(settings):
    async def run():
        session, client = make_session(settings, delay=0.05)
        now = time.time()
        say(session, "them", "i1", "Tell me about a hard bug you fixed.", now - 4, now - 1.0)
        await asyncio.sleep(0.02)  # the first answer has started streaming
        # They kept talking 1 s after finishing the question, while the first answer is still being written.
        say(session, "them", "i2", "Ideally something in production.", now - 0.5, now)
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    [card] = session.cards
    assert card.question == "Tell me about a hard bug you fixed. Ideally something in production."
    assert card.status == "done" and card.text == "I led the migration."
    assert len(client.responses.streams) == 2  # the first run was cancelled and restarted
    assert client.responses.streams[0].closed  # and its HTTP stream closed, so it stops costing tokens


def test_continuation_restarts_even_after_the_first_answer_already_finished(settings):
    async def run():
        session, client = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Tell me about a hard bug you fixed.", now - 3, now - 2.5)
        await settle(session)  # the (incomplete) first answer finishes streaming before they add the rest
        say(session, "them", "i2", "Ideally something in production.", now - 1, now - 0.5)
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    [card] = session.cards
    assert card.question == "Tell me about a hard bug you fixed. Ideally something in production."
    assert card.status == "done" and card.text == "I led the migration."
    assert len(client.responses.streams) == 2  # the finished answer is discarded and regenerated with the full question


def test_answer_gets_a_chinese_digest_automatically(settings):
    async def run():
        session, client = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Can you tell me about your last project?", now - 3, now - 0.1)
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    assert session.cn == "- Participants: Dana"
    cn_call = client.responses.calls[-1]
    assert cn_call["instructions"].startswith("You help a Chinese-speaking user")
    assert "Can you tell me about your last project?" in cn_call["input"]
    assert "I led the migration." in cn_call["input"]


def test_listening_continues_while_an_answer_streams(settings):
    async def run():
        session, _ = make_session(settings, delay=0.05)
        now = time.time()
        say(session, "them", "i1", "What is your biggest strength?", now - 6, now - 3.6)
        await asyncio.sleep(0.03)  # the answer is now streaming
        say(session, "them", "i2", "We value ownership a lot here.", now - 0.5, now)
        say(session, "me", "i3", "Sure, I think ownership is my biggest strength.", now, now + 1)
        await settle(session)
        return session

    session = asyncio.run(run())
    [card] = session.cards  # the later statement neither cancels nor restarts the answer
    assert card.status == "done" and card.question == "What is your biggest strength?"
    assert [s.speaker for s in session.transcript.segments] == ["them", "them", "me"]


def test_new_question_in_same_turn_gets_its_own_card(settings):
    async def run():
        session, _ = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Why do you want this job?", now - 10, now - 7)
        await settle(session)
        say(session, "them", "i2", "And how soon could you start?", now - 4, now - 2)
        await settle(session)
        return session

    session = asyncio.run(run())
    assert [c.question for c in session.cards] == ["Why do you want this job?", "And how soon could you start?"]


def test_same_question_is_not_answered_twice(settings):
    async def run():
        session, _ = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Can you hear me okay?", now - 10, now - 9)
        await settle(session)
        say(session, "me", "i2", "Yes, I can hear you.", now - 8, now - 7)
        say(session, "them", "i3", "Can you hear me okay?", now - 3, now - 1)
        await settle(session)
        return session

    assert len(asyncio.run(run()).cards) == 1


def test_echo_of_them_on_my_mic_is_dropped(settings):
    async def run():
        session, _ = make_session(settings)
        session.auto_answer = False
        now = time.time()
        say(session, "them", "i1", "So tell me about the biggest project you shipped last year.", now - 4, now - 1)
        say(session, "me", "i2", "tell me about the biggest project you shipped last year", now - 3.8, now - 0.9)
        await settle(session)
        return session

    session = asyncio.run(run())
    assert [s.speaker for s in session.transcript.segments] == ["them"]


def test_improve_compares_my_spoken_reply(settings):
    async def run():
        session, client = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "Why should we hire you?", now - 1, now - 0.5)
        await settle(session)
        say(session, "me", "i2", "Because I work hard and learn fast.", time.time(), time.time() + 1)
        await session.command({"type": "improve"})
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    answer, improve = session.cards
    assert improve.kind == "improve" and improve.parent == answer.id and improve.status == "done"
    request = text_of(client.responses.calls[-1]["input"][-1])
    assert "Because I work hard and learn fast." in request and "Missed / weak:" in request


def test_answer_now_uses_the_live_partial(settings):
    async def run():
        session, client = make_session(settings)
        session.on_partial("them", "i9", "How would you design a rate lim")
        await session.command({"type": "answer_now"})
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    [card] = session.cards
    assert card.source == "manual" and "rate lim" in card.question


def test_typed_question_search_and_stop(settings):
    async def run():
        session, client = make_session(settings, delay=0.2)
        await session.command({"type": "ask", "text": "What is Acme's revenue?"})
        await session.command({"type": "search", "card_id": session.cards[0].id})
        await asyncio.sleep(0.05)
        await session.command({"type": "stop", "card_id": session.cards[0].id})
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    typed, web = session.cards
    assert typed.status == "cancelled"
    assert web.kind == "search" and web.status == "done"
    assert [c["tools"] for c in client.responses.calls if "tools" in c] == [[{"type": "web_search"}]]


def test_live_notes_refresh_after_enough_new_segments(settings):
    async def run():
        session, client = make_session(settings)
        session.auto_answer = False
        now = time.time()
        for i in range(8):
            if i % 2:
                say(session, "them", f"i{i}", f"Our roadmap has goal {i} for next quarter.", now + i, now + i + 0.5)
            else:
                say(session, "me", f"i{i}", f"I shipped item {i} last spring with my own team.", now + i, now + i + 0.5)
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    assert session.notes.text == "- Participants: Dana"
    assert len(session.notes.seen) == 8
    notes_call = client.responses.calls[-1]
    assert "goal 7 for next quarter" in notes_call["input"]
    # The next answer carries the refreshed notes after the fixed background.
    assert "Participants: Dana" in session.context("q").notes


def test_new_meeting_saves_and_resets(settings):
    async def run():
        session, _ = make_session(settings)
        await session.command({"type": "ask", "text": "Hello there, who are you?"})
        await settle(session)
        old_id = session.id
        session.id = old_id + "-a"  # ensure a distinct file name within the same second
        await session.command({"type": "new_meeting"})
        return session, old_id + "-a"

    session, old_id = asyncio.run(run())
    assert session.cards == []
    assert (settings.data_dir / "sessions" / f"{old_id}.json").exists()


def test_prepare_writes_a_prep_pack_once_and_reuses_it(settings):
    async def run():
        session, client = make_session(settings)
        session.store.save_profile("acme", {"name": "Acme"})
        session.store.save_doc("acme", "jd.txt", "Senior engineer. Responsibilities: Kafka. Qualifications: Python.")
        first = await session.activate_profile("acme")
        await session.activate_profile("acme")  # material unchanged: the saved pack is reused
        await settle(session)
        return session, client, first

    session, client, info = asyncio.run(run())
    pack_calls = [c for c in client.responses.calls if c.get("instructions", "").startswith("You prepare me")]
    assert len(pack_calls) == 1 and "## Documents: Job description" in pack_calls[0]["input"]
    assert info["summary"] == "prep pack + full text" and "warning" not in info
    assert "## Prep pack\n\n- Participants: Dana" in session.kb.static_text


def test_prepare_still_loads_the_profile_when_the_pack_fails(settings):
    async def run():
        session, _ = make_session(settings)
        session.store.save_profile("acme", {"name": "Acme"})
        session.store.save_doc("acme", "jd.txt", "Kafka role.")

        async def fail(*args, **kwargs):
            raise RuntimeError("rate limited")

        session.answerer.make_pack = fail
        info = await session.activate_profile("acme")
        await settle(session)
        return session, info

    session, info = asyncio.run(run())
    assert "rate limited" in info["warning"] and info["summary"] == "full text"
    assert "Kafka role." in session.kb.static_text


def test_retrieval_uses_the_recent_conversation(settings):
    async def run():
        session, client = make_session(settings)
        session.auto_answer = False
        session.store.save_profile("acme", {"name": "Acme"})
        transcript = "\n".join(f"Interviewer: Tell me about Kafka {i}?\nMe: We ran Kafka at Globex {i}." for i in range(6))
        session.store.save_doc("acme", "round1.txt", transcript + "\n\nInterviewer: What about Redis?\nMe: Redis cache.")
        session.load_profile("acme")
        now = time.time()
        say(session, "them", "i1", "We use Kafka heavily at Acme.", now - 3, now - 2)
        return session.context("Can you go deeper on that?")

    ctx = asyncio.run(run())
    assert ctx.excerpts and "Kafka" in ctx.excerpts[0]


def test_typed_questions_and_the_detail_button_answer_in_depth(settings):
    async def run():
        session, client = make_session(settings)
        now = time.time()
        say(session, "them", "i1", "How would you design a rate limiter?", now - 2, now - 1)
        await settle(session)
        await session.command({"type": "detail", "card_id": session.cards[0].id})
        await session.command({"type": "ask", "text": "What is a token bucket?"})
        await settle(session)
        return session, client

    session, client = asyncio.run(run())
    quick, detailed, typed = session.cards
    assert quick.kind == "answer" and detailed.kind == typed.kind == "detail"
    assert detailed.parent == quick.id and detailed.question == quick.question
    assert detailed.status == typed.status == "done"
    deep_calls = [c for c in client.responses.calls if c.get("stream") and c["model"] == "gpt-6-sol"]
    assert len(deep_calls) == 2
    assert "answer in more depth" in text_of(deep_calls[1]["input"][-1])
    assert "What is a token bucket?" in text_of(deep_calls[1]["input"][-1])
    cn_inputs = [c["input"] for c in client.responses.calls if c.get("instructions", "").startswith("You help a Chinese")]
    assert any("What is a token bucket?" in i for i in cn_inputs)  # detailed answers get the Chinese digest too
