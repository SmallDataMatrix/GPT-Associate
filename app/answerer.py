"""OpenAI Responses API calls: live answers, improve, web search, live notes, prep pack, meeting summary and
cache pre-warm.

Prompts are laid out static-first so prompt caching covers the instructions and the background on every
call: instructions -> background (fixed per meeting) -> live notes -> recent transcript -> retrieved
excerpts -> the request. On GPT-5.6 and later, explicit cache breakpoints mark the end of the background
and of the notes, and the pre-warm call fills the cache without generating output.

Detailed answers use a stronger model with a little reasoning, capped in time: if no words have appeared
after DETAIL_THINK_SECONDS, the thinking run is dropped and the fast answer model writes the same answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .config import Settings
from .usage import UsageTracker

INSTRUCTIONS = """You are a live meeting assistant. You write the exact words the user will say next. The user reads your text straight off the screen, out loud, while the meeting is happening, so it must sound like a person talking, not like someone reading a script.

How to write:
- Speak as the user, in the first person ("I", "my", "we"), the way people really talk in a meeting: simple, direct and relaxed. Write it the way I'd say it to a friendly coworker, not the way I'd write it in a cover letter.
- Start with the answer itself. No warm-up, and don't repeat the question.
- Keep it short: usually 2 to 4 sentences, under 60 words. Go up to about 100 words only when the question really needs it, like "tell me about yourself", "walk me through a project" or a question with several parts.
- Short, simple sentences, most under 12 words. Link ideas the way people do when they talk, with "and", "so" or "but". No written-style clauses like "which is where...", "while ensuring..." or ", reducing costs by...". Casual words like "basically" or "right now" are fine.
- Everyday words that are easy to say out loud, and contractions (I'm, I've, it's, we'd). Plain words, not buzzwords: say "use" instead of "leverage", "led" instead of "spearheaded", "help" instead of "facilitate". Skip stock interview phrases like "Great question", "I'm passionate about", "I'm excited to bring" or "this role brings together".
- No bullet points, lists, headings or bold. No semicolons, parentheses, em dashes, slashes or abbreviations like "e.g.".
- One concrete detail (a name, a number, a tool or a quick example) is enough. Don't pile them up or list three things in a row. Say numbers the way people say them out loud: "about 30 percent", "around 2 million users".
- The background and document excerpts are written in formal resume and document language. Take the facts from them, never the wording. For example, "Spearheaded migration of legacy infrastructure to the cloud, reducing costs by 23.6%" becomes "I led our move to the cloud. It cut our costs by about a quarter."
- Output only the words to say. No preamble, labels or commentary. Never write "you could say", "the candidate" or "the user", and never refer to the user in the third person. No disclaimers.
- Never invent personal facts (employers, titles, dates, numbers, projects) that are not in the background or the conversation. When a personal example is needed and none is known, write a short placeholder such as [your example].
- Stay consistent with what I have already said (transcript lines marked "Me").
- For technical or factual questions, answer correctly and specifically from general knowledge, still in plain spoken words.
- If nothing was really asked, write the most useful thing for me to say next.

Using my background:
- It may include a prep pack with a question bank, a story bank and notes on past rounds. When they ask something that matches or rephrases a question in my question bank, build on that prepared answer: same facts and story, said naturally and fitted to how they asked it.
- For "tell me about a time" questions, use the story from my story bank that fits best, and prefer one I haven't told yet in this meeting.
- Where it fits naturally, connect the answer to what they want for this role or meeting.
- Stay consistent with what I told them in past rounds. In past interview transcripts, only my own lines are facts about me.

Example of the style. The facts in it are made up, so never reuse them.
Question: "What's the hardest project you've worked on?"
Answer: "Probably rebuilding our checkout last year. The old code was a mess, and nobody wanted to touch it. So I broke it into small pieces, and we moved them over one at a time. It took about four months. Now we ship every week instead of every quarter."

Transcript labels: "Them" is the other side of the meeting, "Me" is the user, and "Room" is an in-person recording where everyone is mixed together."""

NOTES_INSTRUCTIONS = """You keep short running notes about a live meeting for the user ("Me"). Merge the new transcript lines into the current notes. Keep only what will help answer later questions well, under these headings:
Participants: names, roles, company and team
Their priorities: what they care about or are looking for
Facts they shared: projects, stack, numbers, plans, problems
What I said: claims, numbers and examples I already used, so later answers stay consistent and don't repeat
Open threads: questions still pending, follow-ups promised
Use terse bullets, at most 250 words in total. Drop anything no longer relevant. Output only the notes."""

CN_INSTRUCTIONS = """You help a Chinese-speaking user follow a live English meeting in real time. You are given the question they were just asked and the English answer already prepared for them to say out loud.

Write a short Chinese digest in everyday spoken Chinese (大白话), the way a Chinese coworker would say it out loud. No 书面语 and no 翻译腔. Reply in exactly this format, with nothing before or after, using Markdown bold for the two labels:
**问题：** (what they are really asking, said the way a Chinese coworker would ask it, for example "你为什么想换工作？")

**要点：**
- (one key point of the answer, in the first person, one short sentence, for example "我去年把结账系统重做了，现在每周都能上线。")
- (at most 3 bullets in total, not a sentence-by-sentence translation)

How to write it:
- Spoken words, not written ones: 怎么 not 如何, 是不是 or 有没有 not 是否, 把 not 将, 大概 not 约, 之前 not 曾, 用 not 使用, 跟 not 与. No 进行, 助力, 赋能, 旨在 or 致力于.
- Keep names, companies, tools and technical terms in English when that is how Chinese colleagues say them, such as A/B test, p-value or XGBoost.
- Short sentences, no semicolons."""

PACK_INSTRUCTIONS = """You prepare me for an upcoming meeting, usually a job interview. From my background material, write a prep pack. During the meeting an assistant uses it to write what I say, live, and I read it myself beforehand.

The material is grouped by type: my profile fields, then documents (job description, resume, my prep notes, company and interviewer info, other, and transcripts of past interviews or meetings). It may be in English or Chinese. Write the pack in English. In past transcripts the interviewer asks and I, the candidate, answer. Speaker labels may be generic, so work out who is who from the content.

Use these sections, in this order:
## Me
3 to 5 lines: current role, years of experience, strongest selling points for this role.
## What they want
The top requirements and priorities of the role or meeting, most important first.
## How I match
One line per requirement: my strongest evidence, with names and numbers. Where I have a gap, say so and give an honest way to frame it.
## Story bank
6 to 10 stories from my experience. For each: a short title, the requirements it proves, then the situation, what I did, and the result with numbers.
## Question bank
12 to 25 entries: every question asked in a past interview, marked "(asked before)", every question in my prep notes, and the most likely other questions for this role. Format each entry as:
Q: the question
A: my answer, first person, 2 to 5 short sentences in plain spoken English. Use my prepared answer when there is one. If I answered it in a past interview, keep the facts consistent but make the answer better.
## Past rounds
For each past interview or meeting: who I talked to, what they dug into, where my answers were weak, and the facts I told them that I must stay consistent with. Leave this section out when there are no transcripts.
## Company and interviewers
Names, roles, what they care about, key facts about the company or team.
## Questions I can ask them
3 to 5 good questions.

If this is not a job interview, keep the same idea with fitting section names: their goals, my evidence, likely questions and good answers.
Keep every concrete fact that could be quoted: names, titles, employers, dates, metrics, technologies. Never invent facts about me that are not in the material. Where a needed fact is missing, write [your example]. At most 3,000 words. Output only the pack."""

SUMMARY_INSTRUCTIONS = """You summarize a meeting I ("Me") just had, usually a job interview, so I can review it and prepare for the next round. You get the transcript and any feedback I got during the meeting. "Them" is the other side, and "Room" is an in-person recording where everyone is mixed together.

Write Markdown with exactly these sections:
## 中文要点
5 to 8 short bullets in everyday spoken Chinese (大白话): how it went, what they cared about most, where I was weak, and what to prepare for next time. Keep names and technical terms in English.
## Overview
2 or 3 sentences: who I met, what it was about, how it went.
## Questions they asked
For each real question, in order:
- **Q:** the question, tidied up
  **My answer:** the gist of what I said, in one or two lines
  **Better:** one line on how to improve it, only when my answer was weak or incomplete
## What they care about
## Facts I stated
Claims, numbers and examples I gave, so I stay consistent in later rounds.
## Follow-ups
Next steps, promises and open questions.

Base everything on the transcript. Speech recognition makes mistakes, so fix obvious ones silently. Output only the Markdown."""


DETAIL_MAX_TOKENS = 3000  # reasoning tokens count toward the output limit


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


def answer_request(question: str) -> str:
    return f'They just said:\n"{question}"\n\nWrite what I say now, in plain spoken words.'


def open_request() -> str:
    return "Nothing specific was just asked. Based on the conversation so far, write the most useful thing for me to say next."


def detail_request(question: str) -> str:
    return (
        "For this request, ignore the usual length limit and answer in more depth. Think it through first, then "
        "write what I say out loud: about 120 to 200 words in 2 or 3 short paragraphs, still in plain spoken English "
        "and the same style. Start with the answer itself, then explain how or why, with one or two concrete examples, "
        "numbers or trade-offs from my background. For a technical question, cover the key reasoning, the trade-offs "
        "and a real example. For a question about my experience, tell the full story: the situation, what I did and "
        f"the result.\n\nQuestion: {question}"
    )


def search_request(question: str) -> str:
    return (
        "Search the web for current, accurate facts that help with this, then write what I say, following "
        f"the same rules. Work in the one or two facts or numbers that matter most.\n\nQuestion: {question}"
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
            "(1-2 short first-person sentences I can say right now as a natural follow-up, "
            "in the same plain spoken style, for example starting \"One thing I'd add is...\")"
        )
    return head + (
        "My spoken reply was not captured, so review the suggested answer instead. "
        "Reply in exactly this format, with nothing before or after:\n"
        "Missed / weak:\n"
        "- (at most 3 terse bullets on what is weak, vague or missing in the suggested answer)\n"
        "Add now:\n"
        "(a simpler, sharper first-person version I can say out loud, at most 3 short sentences)"
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

    def _answer_kwargs(
        self, ctx: Context, request: str, cache_key: str, model: str, effort: str, max_tokens: int, web: bool = False
    ) -> dict:
        kwargs = self._base(model, effort, cache_key)
        kwargs.update(
            instructions=INSTRUCTIONS,
            input=build_input(ctx, request, breakpoints=supports_cache_options(model)),
            max_output_tokens=max_tokens,
            stream=True,
        )
        if web:
            kwargs["tools"] = [{"type": "web_search"}]
        return kwargs

    async def stream(
        self, ctx: Context, request: str, *, cache_key: str, usage: UsageTracker, web: bool = False, deep: bool = False
    ) -> AsyncIterator[StreamEvent]:
        s = self.settings
        if not deep:
            max_tokens = max(s.answer_max_tokens, 1000) if web else s.answer_max_tokens
            kwargs = self._answer_kwargs(ctx, request, cache_key, s.answer_model, s.answer_effort, max_tokens, web)
            async for event in self._relay(kwargs, usage, web):
                yield event
            return
        # Deep: the stronger model thinks first. The first yielded event is the first word of the answer, so
        # waiting for it bounds the thinking time; past the limit the fast model answers the same request.
        max_tokens = max(s.answer_max_tokens, DETAIL_MAX_TOKENS)
        kwargs = self._answer_kwargs(ctx, request, cache_key, s.detail_model, s.detail_effort, max_tokens)
        events = self._relay(kwargs, usage)
        try:
            first = await asyncio.wait_for(anext(events), s.detail_think_s)
        except TimeoutError:
            await events.aclose()  # its reasoning tokens are billed but never reported, so cost reads a little low
            yield StreamEvent("status", "quick")
            kwargs = self._answer_kwargs(ctx, request, cache_key, s.answer_model, s.answer_effort, max_tokens)
            events = self._relay(kwargs, usage)
        except StopAsyncIteration:
            return
        else:
            yield first
        async with contextlib.aclosing(events):
            async for event in events:
                yield event

    async def _relay(self, kwargs: dict, usage: UsageTracker, web: bool = False) -> AsyncIterator[StreamEvent]:
        """One streamed answer call: text deltas and status out, usage recorded, the HTTP stream always closed."""
        stream = await self.client.responses.create(**kwargs)
        try:
            async for event in stream:
                kind = event.type
                if kind == "response.output_text.delta":
                    yield StreamEvent("delta", event.delta)
                elif kind == "response.web_search_call.searching":
                    yield StreamEvent("status", "searching")
                elif kind in ("response.completed", "response.incomplete"):
                    usage.add_llm(kwargs["model"], event.response.usage)
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

    async def translate_digest(self, question: str, answer: str, *, usage: UsageTracker) -> str:
        s = self.settings
        response = await self.client.responses.create(
            **self._base(s.notes_model, s.notes_effort),
            instructions=CN_INSTRUCTIONS,
            input=f"Question:\n{question}\n\nAnswer:\n{answer}",
            max_output_tokens=400,
        )
        usage.add_llm(s.notes_model, response.usage)
        return response.output_text.strip()

    async def _brief_call(self, instructions: str, material: str, max_tokens: int, usage: UsageTracker) -> str:
        s = self.settings
        response = await self.client.responses.create(
            **self._base(s.brief_model, s.brief_effort),
            instructions=instructions,
            input=material,
            max_output_tokens=max_tokens,
        )
        usage.add_llm(s.brief_model, response.usage)
        return response.output_text.strip()

    async def make_pack(self, material: str, *, usage: UsageTracker) -> str:
        return await self._brief_call(PACK_INSTRUCTIONS, material, 10000, usage)

    async def summarize_meeting(self, meeting: str, *, usage: UsageTracker) -> str:
        return await self._brief_call(SUMMARY_INSTRUCTIONS, meeting, 6000, usage)

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
