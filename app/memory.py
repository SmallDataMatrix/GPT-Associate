"""Conversation memory: the live transcript and the running notes that refresh during the meeting."""

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
