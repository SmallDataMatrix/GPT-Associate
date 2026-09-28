"""OpenAI Responses API calls: live answers, improve, web search, live notes, brief and cache pre-warm.

Prompts are laid out static-first so prompt caching covers the instructions and the background on every
call: instructions -> background (fixed per meeting) -> live notes -> recent transcript -> retrieved
excerpts -> the request. On GPT-5.6 and later, explicit cache breakpoints mark the end of the background
and of the notes, and the pre-warm call fills the cache without generating output.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .config import Settings
from .usage import UsageTracker

INSTRUCTIONS = """You are a live meeting assistant. You write the exact words the user will say next. The user reads your text straight off the screen, out loud, while the meeting is happening.

How to write:
- Speak as the user, in the first person ("I", "my", "we"), in natural spoken English.
- Open with the direct answer in one or two sentences.
- Then add at most 3 short bullet points ("- ") with concrete specifics: names, numbers, tools, examples. Take them from the background and the conversation.
- Stay under 120 words unless the question clearly needs more.
- Output only the words to say. No preamble, labels, headings or commentary. Never write "you could say", "the candidate" or "the user", and never refer to the user in the third person. No disclaimers.
- Never invent personal facts (employers, titles, dates, numbers, projects) that are not in the background or the conversation. When a personal example is needed and none is known, write a short placeholder such as [your example].
- Stay consistent with what I have already said (transcript lines marked "Me").
- For technical or factual questions, answer correctly and specifically from general knowledge.
- If nothing was really asked, write the most useful thing for me to say next.

Transcript labels: "Them" is the other side of the meeting, "Me" is the user, and "Room" is an in-person recording where everyone is mixed together."""

NOTES_INSTRUCTIONS = """You keep short running notes about a live meeting for the user ("Me"). Merge the new transcript lines into the current notes. Keep only what will help answer later questions well, under these headings:
Participants: names, roles, company and team
Their priorities: what they care about or are looking for
Facts they shared: projects, stack, numbers, plans, problems
What I said: claims, numbers and examples I already used, so later answers stay consistent and don't repeat
Open threads: questions still pending, follow-ups promised
Use terse bullets, at most 250 words in total. Drop anything no longer relevant. Output only the notes."""

BRIEF_INSTRUCTIONS = """Condense the background material into a briefing for an assistant that writes answers in my voice during a meeting. Keep every concrete fact that could be quoted in an answer: names, titles, employers, dates, metrics, technologies, projects, achievements, and the requirements of the role or meeting. Use these sections:
## Me
## Experience and projects (bullets with metrics)
## Stories (short situation, action, result examples)
## The role, meeting and other side
## Likely questions and my best answers (5 to 10, first person)
At most 1,500 words. Output only the briefing."""


@dataclass
class Context:
    background: str = ""
    notes: str = ""
    transcript: str = ""
    excerpts: list[str] = field(default_factory=list)


@dataclass
class StreamEvent:
    kind: str  # delta | status
    text: str = ""


def supports_cache_options(model: str) -> bool:
    """Explicit cache breakpoints and no-output pre-warming exist on GPT-5.6 and later."""
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", model)
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (5, 6)


def _content(text: str, breakpoint: bool = False) -> list[dict]:
    block: dict = {"type": "input_text", "text": text}
    if breakpoint:
        block["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return [block]


def build_input(ctx: Context, request: str, breakpoints: bool = False) -> list[dict]:
    background = ctx.background.strip() or "(No background was provided.)"
    items = [{"role": "developer", "content": _content(f"# My background (fixed for this meeting)\n\n{background}", breakpoints)}]
    if ctx.notes:
        items.append({"role": "developer", "content": _content(ctx.notes, breakpoints)})
    conversation = "# Conversation so far\n\n" + (ctx.transcript or "(Nothing has been said yet.)")
    if ctx.excerpts:
        conversation += "\n\n# Relevant excerpts from my documents\n\n" + "\n\n---\n\n".join(ctx.excerpts)
    items.append({"role": "developer", "content": _content(conversation)})
    items.append({"role": "user", "content": _content(request)})
    return items


def answer_request(question: str, typed: bool = False) -> str:
    if typed:
        return f"Question: {question}\n\nWrite what I say."
    return f'They just said:\n"{question}"\n\nWrite what I say now.'


def open_request() -> str:
    return "Nothing specific was just asked. Based on the conversation so far, write the most useful thing for me to say next."


def search_request(question: str) -> str:
    return (
        "Search the web for current, accurate facts that help with this, then write what I say, following "
        f"the same rules. Include the key facts and numbers.\n\nQuestion: {question}"
    )


def improve_request(question: str, suggested: str, my_reply: str) -> str:
    head = (
        "For this request, use the format below instead of the normal answer format.\n\n"
        f'The question was:\n"{question}"\n\n'
        f'The suggested answer on my screen was:\n"{suggested.strip() or "(none)"}"\n\n'
    )
    if my_reply.strip():
        return head + (
            f'What I actually said:\n"{my_reply.strip()}"\n\n'
            "Compare what I actually said with the question, my background and the conversation. "
            "Reply in exactly this format, with nothing before or after:\n"
            "Missed / weak:\n"
            "- (at most 3 terse bullets: points I left out, got wrong, or said vaguely)\n"
            "Add now:\n"
            "(1-2 first-person sentences I can say right now as a natural follow-up, "
            "for example starting \"One thing I'd add is...\")"
        )
    return head + (
        "My spoken reply was not captured, so review the suggested answer instead. "
        "Reply in exactly this format, with nothing before or after:\n"
        "Missed / weak:\n"
        "- (at most 3 terse bullets on what is weak, vague or missing in the suggested answer)\n"
        "Add now:\n"
        "(a sharper first-person version, at most 3 sentences)"
    )


class Answerer:
    def __init__(self, client, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    @property
    def available(self) -> bool:
        return self.client is not None

    def _base(self, model: str, effort: str, cache_key: str = "") -> dict:
        kwargs: dict = {"model": model, "store": False}
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key
        return kwargs

    async def stream(
        self, ctx: Context, request: str, *, cache_key: str, usage: UsageTracker, web: bool = False
    ) -> AsyncIterator[StreamEvent]:
        s = self.settings
        kwargs = self._base(s.answer_model, s.answer_effort, cache_key)
        kwargs.update(
            instructions=INSTRUCTIONS,
            input=build_input(ctx, request, breakpoints=supports_cache_options(s.answer_model)),
            max_output_tokens=max(s.answer_max_tokens, 1000) if web else s.answer_max_tokens,
            stream=True,
        )
        if web:
            kwargs["tools"] = [{"type": "web_search"}]
        stream = await self.client.responses.create(**kwargs)
        try:
            async for event in stream:
                kind = event.type
                if kind == "response.output_text.delta":
                    yield StreamEvent("delta", event.delta)
                elif kind == "response.web_search_call.searching":
                    yield StreamEvent("status", "searching")
                elif kind in ("response.completed", "response.incomplete"):
                    usage.add_llm(s.answer_model, event.response.usage)
                    if web:
                        usage.add_search()
                elif kind == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise RuntimeError(getattr(error, "message", None) or "The answer request failed.")
                elif kind == "error":
                    raise RuntimeError(getattr(event, "message", None) or "The answer stream failed.")
        finally:
            await stream.close()

    async def update_notes(self, notes: str, new_lines: str, *, usage: UsageTracker) -> str:
        s = self.settings
        response = await self.client.responses.create(
            **self._base(s.notes_model, s.notes_effort),
            instructions=NOTES_INSTRUCTIONS,
            input=f"Current notes:\n{notes or '(empty)'}\n\nNew transcript lines:\n{new_lines}",
            max_output_tokens=800,
        )
        usage.add_llm(s.notes_model, response.usage)
        return response.output_text.strip() or notes

    async def make_brief(self, material: str, *, usage: UsageTracker) -> str:
        s = self.settings
        response = await self.client.responses.create(
            **self._base(s.brief_model, s.brief_effort),
            instructions=BRIEF_INSTRUCTIONS,
            input=material,
            max_output_tokens=8000,
        )
        usage.add_llm(s.brief_model, response.usage)
        return response.output_text.strip()

    async def prewarm(self, ctx: Context, *, cache_key: str, usage: UsageTracker) -> None:
        """Opens the HTTP connection and writes the instructions + background prefix into the prompt cache."""
        s = self.settings
        modern = supports_cache_options(s.answer_model)
        kwargs = self._base(s.answer_model, s.answer_effort, cache_key)
        if modern:
            kwargs["prompt_cache_options"] = {"prewarm": True}
        response = await self.client.responses.create(
            **kwargs,
            instructions=INSTRUCTIONS,
            input=build_input(ctx, "Reply with OK.", breakpoints=modern),
            max_output_tokens=16,
        )
        usage.add_llm(s.answer_model, getattr(response, "usage", None))
