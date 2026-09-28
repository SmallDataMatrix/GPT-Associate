"""The live meeting: transcript events in, answer cards out to every connected screen.

Transcription, answer generation and live-notes updates run as independent asyncio tasks, so listening
never pauses while an answer is being written.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from .answerer import Answerer, Context, answer_request, improve_request, open_request, search_request
from .config import Settings
from .detector import is_echo, is_question, normalize
from .knowledge import KnowledgeBase, build_knowledge, needs_brief, source_hash, source_material
from .memory import LiveNotes, Segment, Transcript
from .store import Store
from .transcriber import Transcriber
from .usage import UsageTracker

log = logging.getLogger(__name__)

ASKING = {"them", "room"}  # sources whose questions get answered
TURN_GAP_S = 5.0  # silence that ends the other side's turn
CONTINUE_WINDOW_S = 2.5  # speech resuming this soon after the question restarts its answer with the full question
DEDUPE_WINDOW_S = 10.0
ECHO_WINDOW_S = 8.0
MAX_QUESTION_CHARS = 800
TRANSCRIPT_PROMPT_CHARS = 12000  # about 3k tokens of recent conversation per answer
PREWARM_EVERY_S = 240.0
TICK_S = 10.0

_card_ids = itertools.count(1)


@dataclass
class Card:
    kind: str  # answer | search | improve
    source: str  # auto | manual | typed | search | improve
    question: str
    request: str
    t0: float  # end of the question (or the button press): latency is measured from here
    id: str = field(default_factory=lambda: f"c{next(_card_ids)}")
    created: float = field(default_factory=time.time)
    started: float = field(default_factory=time.time)
    text: str = ""
    status: str = "pending"  # pending | searching | streaming | done | cancelled | error
    first_ms: int | None = None
    done_ms: int | None = None
    error: str = ""
    parent: str = ""
    turn_no: int = -1
    turn_start: int = 0
    turn_end: int = 0
    gen: int = 0  # bumped on restart so a cancelled run can't overwrite the new one

    def public(self) -> dict:
        keys = ("id", "kind", "source", "question", "created", "text", "status", "first_ms", "done_ms", "error", "parent")
        return {k: getattr(self, k) for k in keys}


class _Viewer:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=5000)

    async def pump(self) -> None:
        try:
            while True:
                event = await self.queue.get()
                await self.ws.send_json(event)
        except Exception:
            pass  # the page went away; serve() cleans up


class Hub:
    """Fans events out to every open page, in order, without letting a slow phone block the meeting."""

    def __init__(self) -> None:
        self._viewers: set[_Viewer] = set()

    def publish(self, event: dict) -> None:
        for viewer in list(self._viewers):
            try:
                viewer.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Too far behind: drop it; the page reconnects and gets a fresh snapshot.
                self._viewers.discard(viewer)
                asyncio.ensure_future(self._close(viewer.ws))

    @staticmethod
    async def _close(ws: WebSocket) -> None:
        with contextlib.suppress(Exception):
            await ws.close(code=1013)

    async def serve(self, ws: WebSocket, snapshot: dict, on_command) -> None:
        viewer = _Viewer(ws)
        viewer.queue.put_nowait(snapshot)
        self._viewers.add(viewer)
        writer = asyncio.create_task(viewer.pump())
        try:
            while True:
                try:
                    message = await ws.receive_json()
                except (WebSocketDisconnect, RuntimeError):
                    break
                except ValueError:
                    continue
                try:
                    await on_command(message)
                except Exception:
                    log.exception("command failed: %s", message)
        finally:
            self._viewers.discard(viewer)
            writer.cancel()


class MeetingSession:
    def __init__(self, settings: Settings, store: Store, answerer: Answerer, transcriber_factory=Transcriber) -> None:
        self.settings = settings
        self.store = store
        self.answerer = answerer
        self.transcriber_factory = transcriber_factory
        self.hub = Hub()
        self.kb = KnowledgeBase()
        self.profile: dict = {"slug": "", "name": ""}
        self.auto_answer = True
        self.transcribers: dict[str, Transcriber] = {}
        self.stt_status: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._background: set[asyncio.Task] = set()
        self._last_warm = 0.0
        self._reset()

    def _reset(self) -> None:
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.started = time.time()
        self.transcript = Transcript()
        self.notes = LiveNotes()
        self.usage = UsageTracker()
        self.cards: list[Card] = []
        self.partials: dict[str, dict] = {}  # item_id -> {"speaker", "text"} while speech is being transcribed
        self.item_times: dict[str, dict] = {}  # item_id -> {"start", "stop"} from VAD events
        self.turn: list[Segment] = []  # the other side's current turn
        self.turn_no = 0
        self._last_question = ("", 0.0)
        self._dirty = False
        self._usage_sent = None

    # Helpers ---------------------------------------------------------------------

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _publish_card(self, card: Card) -> None:
        self.hub.publish({"type": "card", "card": card.public()})

    def _publish_usage(self) -> None:
        data = self.usage.public()
        if data != self._usage_sent:
            self._usage_sent = data
            self.hub.publish({"type": "usage", "usage": data})

    @property
    def cache_key(self) -> str:
        return f"ga-{self.profile.get('slug') or 'default'}"

    def context(self, question: str) -> Context:
        return Context(
            background=self.kb.static_text,
            notes=self.notes.render(),
            transcript=self.transcript.render(max_chars=TRANSCRIPT_PROMPT_CHARS),
            excerpts=self.kb.retrieve(question),
        )

    def profile_info(self) -> dict:
        return {**self.profile, "mode": self.kb.mode, "tokens": self.kb.tokens}

    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "session": self.id,
            "transcript": [s.public() for s in self.transcript.segments[-400:]],
            "partials": [{"id": k, **v} for k, v in self.partials.items()],
            "cards": [c.public() for c in self.cards[-60:]],
            "notes": self.notes.public(),
            "auto": self.auto_answer,
            "usage": self.usage.public(),
            "profile": self.profile_info(),
            "stt": self.stt_status,
            "models": {"answer": self.settings.answer_model, "transcribe": self.settings.transcribe_model},
        }

    # Transcript events (called by Transcriber) -----------------------------------

    def on_speech_started(self, source: str, item_id: str) -> None:
        self.item_times.setdefault(item_id, {})["start"] = time.time()

    def on_speech_stopped(self, source: str, item_id: str) -> None:
        self.item_times.setdefault(item_id, {})["stop"] = time.time()

    def on_partial(self, source: str, item_id: str, delta: str) -> None:
        partial = self.partials.setdefault(item_id, {"speaker": source, "text": ""})
        partial["text"] += delta
        self.hub.publish({"type": "partial", "id": item_id, "speaker": source, "text": partial["text"]})

    def on_final(self, source: str, item_id: str, text: str) -> None:
        self.partials.pop(item_id, None)
        times = self.item_times.pop(item_id, {})
        text = " ".join(text.split())
        now = time.time()
        start = times.get("start", now)
        if not text or (source == "me" and self._is_echo(text, start)):
            self.hub.publish({"type": "partial_drop", "id": item_id})
            return
        seg = Segment(item_id, source, text, start, times.get("stop", now))
        self.transcript.add(seg)
        self._dirty = True
        self.hub.publish({"type": "segment", "segment": seg.public()})
        if source in ASKING:
            # Latency is measured from the real end of speech, before VAD's silence window.
            self._on_asking_segment(seg, seg.t_end - self.settings.vad_silence_ms / 1000)
        else:
            self._end_turn()
        self._maybe_update_notes()

    def on_stt_status(self, source: str, status: str) -> None:
        self.stt_status[source] = status
        self.hub.publish({"type": "stt", "stt": self.stt_status})

    def on_stt_error(self, source: str, message: str) -> None:
        self.hub.publish({"type": "toast", "level": "error", "message": f"{source}: {message}"})

    def on_audio_seconds(self, source: str, seconds: float) -> None:
        self.usage.add_audio(self.settings.transcribe_model, seconds)

    def _is_echo(self, text: str, start: float) -> bool:
        theirs = [s.text for s in self.transcript.segments[-20:] if s.speaker == "them" and s.t_end >= start - ECHO_WINDOW_S]
        theirs += [p["text"] for p in self.partials.values() if p["speaker"] == "them"]
        return is_echo(text, theirs)

    # Question handling -----------------------------------------------------------

    def _end_turn(self) -> None:
        if self.turn:
            self.turn = []
            self.turn_no += 1

    def _on_asking_segment(self, seg: Segment, t0: float) -> None:
        if self.turn and seg.t - self.turn[-1].t_end > TURN_GAP_S:
            self._end_turn()
        self.turn.append(seg)
        if not self.auto_answer:
            return
        latest = next((c for c in reversed(self.cards) if c.source == "auto"), None)
        same_turn = latest is not None and latest.turn_no == self.turn_no
        if same_turn and latest.id in self._tasks and seg.t - latest.t0 <= CONTINUE_WINDOW_S:
            self._restart(latest, t0)  # they kept talking right after the question: answer the whole thing
        elif is_question(seg.text):
            self._start_auto(latest.turn_end if same_turn else 0, t0)

    def _turn_text(self, start: int) -> str:
        return " ".join(s.text for s in self.turn[start:])[-MAX_QUESTION_CHARS:]

    def _start_auto(self, start: int, t0: float) -> None:
        question = self._turn_text(start)
        key, when = self._last_question
        if normalize(question) == key and time.time() - when < DEDUPE_WINDOW_S:
            return
        self._last_question = (normalize(question), time.time())
        card = Card("answer", "auto", question, answer_request(question), t0,
                    turn_no=self.turn_no, turn_start=start, turn_end=len(self.turn))
        self._launch(card)

    def _restart(self, card: Card, t0: float) -> None:
        self._cancel(card.id)
        card.question = self._turn_text(card.turn_start)
        card.request = answer_request(card.question)
        card.turn_end = len(self.turn)
        card.text, card.status, card.error = "", "pending", ""
        card.first_ms = card.done_ms = None
        card.t0, card.started = t0, time.time()
        card.gen += 1
        self._last_question = (normalize(card.question), time.time())
        self._publish_card(card)
        self._run(card)

    def _launch(self, card: Card) -> None:
        self.cards.append(card)
        self._dirty = True
        self._publish_card(card)
        self._run(card)

    def _run(self, card: Card) -> None:
        if not self.answerer.available:
            card.status, card.error = "error", "No OpenAI API key configured."
            self._publish_card(card)
            return
        task = asyncio.create_task(self._answer(card, card.gen))
        self._tasks[card.id] = task

        def _done(t: asyncio.Task, card_id: str = card.id) -> None:
            if self._tasks.get(card_id) is t:
                del self._tasks[card_id]

        task.add_done_callback(_done)

    def _cancel(self, card_id: str) -> None:
        task = self._tasks.pop(card_id, None)
        if task:
            task.cancel()

    async def _answer(self, card: Card, gen: int) -> None:
        def current() -> bool:
            return card.gen == gen

        try:
            stream = self.answerer.stream(
                self.context(card.question), card.request,
                cache_key=self.cache_key, usage=self.usage, web=card.kind == "search",
            )
            async with contextlib.aclosing(stream):
                async for event in stream:
                    if not current():
                        return
                    if event.kind == "status":
                        card.status = event.text
                        self._publish_card(card)
                    elif event.kind == "delta" and event.text:
                        card.text += event.text
                        if card.first_ms is None:
                            card.first_ms = int((time.time() - card.t0) * 1000)
                            card.status = "streaming"
                            self._publish_card(card)
                        else:
                            self.hub.publish({"type": "delta", "id": card.id, "d": event.text})
            if current():
                card.status = "done"
                card.done_ms = int((time.time() - card.t0) * 1000)
        except asyncio.CancelledError:
            if current() and card.status not in ("done", "error"):
                card.status = "cancelled"
                self._publish_card(card)
            raise
        except Exception as exc:
            log.exception("answer failed")
            if current():
                card.status, card.error = "error", str(exc)
        if current():
            self._dirty = True
            self._publish_card(card)
            self._publish_usage()

    # Commands from the page ------------------------------------------------------

    async def command(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "ask":
            text = str(message.get("text", "")).strip()
            if text:
                self._launch(Card("answer", "typed", text, answer_request(text, typed=True), time.time()))
        elif kind == "answer_now":
            self.answer_now()
        elif kind == "improve":
            self.improve(message.get("card_id"))
        elif kind == "search":
            self.search(message.get("card_id"))
        elif kind == "stop":
            card = self._find(message.get("card_id"))
            if card:
                self._cancel(card.id)
                if card.status not in ("done", "error"):
                    card.status = "cancelled"
                    self._publish_card(card)
        elif kind == "auto":
            self.auto_answer = bool(message.get("value"))
            self.hub.publish({"type": "auto", "value": self.auto_answer})
        elif kind == "note":
            text = str(message.get("text", "")).strip()
            if text:
                self.notes.manual.append(text)
                self._dirty = True
                self.hub.publish({"type": "notes", "notes": self.notes.public()})
        elif kind == "new_meeting":
            await self.new_meeting()

    def _find(self, card_id) -> Card | None:
        return next((c for c in self.cards if c.id == card_id), None)

    def answer_now(self) -> None:
        segments = list(self.turn) or self.transcript.last_run(ASKING)
        partial = " ".join(p["text"].strip() for p in self.partials.values() if p["speaker"] in ASKING)
        question = " ".join(s.text for s in segments) + (" " + partial if partial else "")
        question = question.strip()[-MAX_QUESTION_CHARS:]
        if question:
            card = Card("answer", "manual", question, answer_request(question), time.time(), turn_no=self.turn_no)
        else:
            card = Card("answer", "manual", "(what to say next)", open_request(), time.time())
        self._launch(card)

    def _target(self, card_id) -> Card | None:
        card = self._find(card_id) if card_id else None
        return card or next((c for c in reversed(self.cards) if c.kind in ("answer", "search")), None)

    def improve(self, card_id=None) -> None:
        target = self._target(card_id)
        if not target:
            self.hub.publish({"type": "toast", "level": "info", "message": "No answer to improve yet."})
            return
        since = target.created - 1
        mine = [s for s in self.transcript.segments if s.speaker == "me" and s.t >= since]
        if not mine:  # in person there is one mixed stream: take everything said after the question
            mine = [s for s in self.transcript.segments if s.speaker == "room" and s.t >= target.created]
        partial = " ".join(p["text"].strip() for p in self.partials.values() if p["speaker"] in ("me", "room"))
        reply = " ".join(s.text for s in mine) + (" " + partial if partial else "")
        request = improve_request(target.question, target.text, reply)
        self._launch(Card("improve", "improve", target.question, request, time.time(), parent=target.id))

    def search(self, card_id=None) -> None:
        target = self._target(card_id)
        if not target:
            return
        self._launch(Card("search", "search", target.question, search_request(target.question), time.time(), parent=target.id))

    async def new_meeting(self) -> None:
        self.save()
        for card_id in list(self._tasks):
            self._cancel(card_id)
        self._reset()
        self.hub.publish(self.snapshot())

    # Live notes, background, persistence -----------------------------------------

    def _maybe_update_notes(self) -> None:
        if self.answerer.available and self.notes.due(self.transcript, time.time()):
            self.notes.running = True
            self._spawn(self._update_notes())

    async def _update_notes(self) -> None:
        new = self.notes.new_segments(self.transcript)
        try:
            self.notes.text = await self.answerer.update_notes(
                self.notes.text, self.transcript.render(new, max_chars=20000), usage=self.usage
            )
            self.notes.mark(new, time.time())
            self._dirty = True
            self.hub.publish({"type": "notes", "notes": self.notes.public()})
            self._publish_usage()
        except Exception:
            log.exception("live notes update failed")
            self.notes.last_update = time.time()  # back off until the next interval
        finally:
            self.notes.running = False

    def prewarm(self, force: bool = False) -> None:
        if not self.answerer.available or (not force and time.time() - self._last_warm < PREWARM_EVERY_S):
            return
        self._last_warm = time.time()

        async def warm() -> None:
            try:
                await self.answerer.prewarm(self.context(""), cache_key=self.cache_key, usage=self.usage)
                self._publish_usage()
            except Exception as exc:
                log.warning("cache pre-warm failed: %s", exc)
                self.hub.publish({"type": "toast", "level": "error", "message": f"OpenAI check failed: {exc}"})

        self._spawn(warm())

    def load_profile(self, slug: str, brief: str | None = None) -> None:
        profile = self.store.load_profile(slug)
        docs = self.store.load_docs(slug)
        if brief is None:
            fresh = profile.get("brief_hash") == source_hash(profile, docs)
            brief = profile.get("brief", "") if fresh else ""
        self.kb = build_knowledge(profile, docs, self.settings.full_text_token_limit, brief)
        self.profile = {"slug": slug, "name": profile["name"]}

    async def activate_profile(self, slug: str) -> dict:
        profile = self.store.load_profile(slug)
        docs = self.store.load_docs(slug)
        digest = source_hash(profile, docs)
        brief = profile.get("brief", "") if profile.get("brief_hash") == digest else ""
        if not brief and self.answerer.available and needs_brief(profile, docs, self.settings.full_text_token_limit):
            brief = await self.answerer.make_brief(source_material(profile, docs), usage=self.usage)
            self.store.save_profile(slug, {"brief": brief, "brief_hash": digest})
        self.load_profile(slug, brief)
        self.store.set_active(slug)
        for transcriber in self.transcribers.values():
            await transcriber.set_terms(self.kb.terms)
        self.prewarm(force=True)
        info = self.profile_info()
        self.hub.publish({"type": "profile", "profile": info})
        self._publish_usage()
        return info

    def attach_audio(self, source: str) -> Transcriber:
        old = self.transcribers.pop(source, None)
        if old:
            self._spawn(old.stop())
        transcriber = self.transcriber_factory(source, self.settings, self, terms=self.kb.terms)
        self.transcribers[source] = transcriber
        transcriber.start()
        self.prewarm()
        return transcriber

    async def detach_audio(self, source: str, transcriber: Transcriber) -> None:
        if self.transcribers.get(source) is transcriber:
            del self.transcribers[source]
            self.stt_status.pop(source, None)
            self.hub.publish({"type": "stt", "stt": self.stt_status})
        await transcriber.stop()

    def save(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.store.save_session(self.id, {
            "id": self.id,
            "started": self.started,
            "profile": self.profile,
            "transcript": [s.public() | {"t_end": s.t_end} for s in self.transcript.segments],
            "cards": [c.public() for c in self.cards],
            "notes": self.notes.public(),
            "usage": self.usage.public(),
        })

    async def run_ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                self._maybe_update_notes()
                self._publish_usage()
                self.save()
            except Exception:
                log.exception("periodic tick failed")

    async def shutdown(self) -> None:
        for card_id in list(self._tasks):
            self._cancel(card_id)
        for transcriber in list(self.transcribers.values()):
            await transcriber.stop()
        self.save()
