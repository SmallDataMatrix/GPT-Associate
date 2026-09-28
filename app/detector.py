"""Free, instant heuristics: does a transcript segment ask something, and is a mic segment an echo."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

QUESTION_STARTS = frozenset(
    """what what's whats why how how's hows when where where's who who's whom whose which can can't could
    couldn't would wouldn't will won't should shall do don't does doesn't did didn't is isn't are aren't
    was were have haven't has had may might anything any""".split()
)
PROMPT_PHRASES = (
    "tell me", "tell us", "walk me through", "walk us through", "talk me through", "talk us through",
    "take me through", "take us through", "give me an example", "give us an example", "share with me",
    "share with us", "an example of", "how would you", "how do you", "what would you", "i'd like to hear",
    "i would like to hear", "i'd love to hear", "i'm curious", "i am curious", "curious about", "can you",
    "could you", "would you", "do you have", "what about", "how about", "thoughts on", "your take on",
    "why did you", "why do you", "what made you", "let's talk about your", "i'd like to know",
    "i would like to know", "wondering",
)
# Imperatives that ask me to speak when they open a sentence ("Describe a time...", "Explain how...").
IMPERATIVE_STARTS = frozenset(
    "describe explain tell walk talk give share elaborate expand name list imagine suppose compare "
    "define outline summarize summarise".split()
)
FILLERS = frozenset(
    "so okay ok alright and well um uh right great cool now next then yeah yes sure perfect thanks "
    "fine good nice awesome also".split()
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def normalize(text: str) -> str:
    return " ".join(words(text))


def is_question(text: str, min_words: int = 4) -> bool:
    if len(words(text)) < min_words:
        return False
    if "?" in text:
        return True
    lowered = normalize(text)
    if any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in PROMPT_PHRASES):
        return True
    for sentence in re.split(r"[.!;:]+", text):
        w = words(sentence)
        i = 0
        while i < len(w) - 1 and w[i] in FILLERS:
            i += 1
        if w and (w[i] in QUESTION_STARTS or w[i] in IMPERATIVE_STARTS):
            return True
    return False


def is_echo(mine: str, theirs: list[str], threshold: float = 0.8) -> bool:
    """True when a mic segment is just the other side's audio picked up from the speakers."""
    a = normalize(mine)
    if not a:
        return False
    short = len(a.split()) < 3
    for text in theirs:
        b = normalize(text)
        if not b:
            continue
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        if short:
            if matcher.ratio() >= 0.9:
                return True
            continue
        if matcher.ratio() >= threshold:
            return True
        matched = sum(block.size for block in matcher.get_matching_blocks())
        if matched / len(a) >= threshold:
            return True
    return False
