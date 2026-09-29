"""Conversation memory: the live transcript, the running notes, and saved meetings rendered as Markdown."""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field

SPEAKER_LABELS = {"them": "Them", "me": "Me", "room": "Room"}
NOTES_EVERY_SEGMENTS = 8
NOTES_EVERY_SECONDS = 90.0


@dataclass
class Segment:
    id: str
    speaker: str  # them | me | room
    text: str
    t: float  # speech start, epoch seconds
    t_end: float  # speech stop, epoch seconds

    def public(self) -> dict:
        return {"id": self.id, "speaker": self.speaker, "text": self.text, "t": self.t}


class Transcript:
    def __init__(self) -> None:
        self.segments: list[Segment] = []

    def add(self, seg: Segment) -> None:
        # Completion events can arrive slightly out of order; keep segments ordered by speech start.
        bisect.insort(self.segments, seg, key=lambda s: s.t)

    def render(self, segments: list[Segment] | None = None, max_chars: int = 12000) -> str:
        runs: list[tuple[str, list[str]]] = []
        for seg in self.segments if segments is None else segments:
            if runs and runs[-1][0] == seg.speaker:
                runs[-1][1].append(seg.text)
            else:
                runs.append((seg.speaker, [seg.text]))
        lines: list[str] = []
        total = 0
        for speaker, texts in reversed(runs):
            line = f"{SPEAKER_LABELS.get(speaker, speaker)}: {' '.join(texts)}"
            if total + len(line) > max_chars:
                if not lines:
                    lines.append(line[-max_chars:])
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(reversed(lines))

    def last_run(self, speakers: set[str]) -> list[Segment]:
        """The most recent run of consecutive segments from any of these speakers."""
        run: list[Segment] = []
        for seg in reversed(self.segments):
            if seg.speaker in speakers:
                run.append(seg)
            elif run:
                break
        return list(reversed(run))


@dataclass
class LiveNotes:
    text: str = ""
    seen: set[str] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)
    running: bool = False

    def new_segments(self, transcript: Transcript) -> list[Segment]:
        return [s for s in transcript.segments if s.id not in self.seen]

    def due(self, transcript: Transcript, now: float) -> bool:
        if self.running:
            return False
        pending = len(self.new_segments(transcript))
        return pending >= NOTES_EVERY_SEGMENTS or (pending > 0 and now - self.last_update >= NOTES_EVERY_SECONDS)

    def mark(self, segments: list[Segment], now: float) -> None:
        self.seen.update(s.id for s in segments)
        self.last_update = now

    def render(self) -> str:
        return f"# Live notes from this meeting\n\n{self.text}" if self.text else ""

    def public(self) -> dict:
        return {"text": self.text}


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def meeting_markdown(session: dict, summary: str = "") -> str:
    """A saved meeting as Markdown: the summary, feedback from Improve, then the labelled transcript."""
    started = session.get("started") or 0
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(started)) if started else session.get("id", "")
    profile = (session.get("profile") or {}).get("name") or ""
    lines = [f"# Meeting {when}" + (f" ({profile})" if profile else "")]
    if summary.strip():
        lines += ["", summary.strip()]
    feedback = [c for c in session.get("cards") or [] if c.get("kind") == "improve" and (c.get("text") or "").strip()]
    if feedback:
        lines += ["", "## Feedback during the meeting"]
        for card in feedback:
            lines += ["", f"**Q:** {card.get('question', '')}", "", card["text"].strip()]
    lines += ["", "## Transcript", ""]
    segments = session.get("transcript") or []
    origin = started or (segments[0].get("t") or 0 if segments else 0)
    runs: list[tuple[str, float, list[str]]] = []
    for seg in segments:
        if runs and runs[-1][0] == seg.get("speaker"):
            runs[-1][2].append(seg.get("text", ""))
        else:
            runs.append((seg.get("speaker", ""), seg.get("t") or origin, [seg.get("text", "")]))
    for speaker, t, texts in runs:
        lines.append(f"**{SPEAKER_LABELS.get(speaker, speaker)}** [{_clock(t - origin)}]: {' '.join(texts)}")
        lines.append("")
    if not runs:
        lines.append("(Nothing was transcribed.)")
    return "\n".join(lines).rstrip() + "\n"
