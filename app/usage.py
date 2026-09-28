"""Token, audio and search usage for the current meeting, with an estimated cost."""

from __future__ import annotations

from .config import LLM_PRICES, STT_PRICES, WEB_SEARCH_PRICE


class UsageTracker:
    def __init__(self) -> None:
        self.llm: dict[str, dict[str, int]] = {}
        self.audio_seconds: dict[str, float] = {}
        self.searches = 0

    def add_llm(self, model: str, usage) -> None:
        if usage is None:
            return
        entry = self.llm.setdefault(model, {"input": 0, "cached": 0, "output": 0, "calls": 0})
        entry["input"] += getattr(usage, "input_tokens", 0) or 0
        details = getattr(usage, "input_tokens_details", None)
        entry["cached"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0
        entry["output"] += getattr(usage, "output_tokens", 0) or 0
        entry["calls"] += 1

    def add_audio(self, model: str, seconds: float) -> None:
        self.audio_seconds[model] = self.audio_seconds.get(model, 0.0) + seconds

    def add_search(self) -> None:
        self.searches += 1

    def cost(self) -> float:
        total = self.searches * WEB_SEARCH_PRICE
        for model, e in self.llm.items():
            price_in, price_cached, price_out = LLM_PRICES.get(model, (0.0, 0.0, 0.0))
            uncached = e["input"] - e["cached"]
            total += (uncached * price_in + e["cached"] * price_cached + e["output"] * price_out) / 1_000_000
        for model, seconds in self.audio_seconds.items():
            total += seconds / 60 * STT_PRICES.get(model, 0.0)
        return total

    def public(self) -> dict:
        tokens_in = sum(e["input"] for e in self.llm.values())
        cached = sum(e["cached"] for e in self.llm.values())
        return {
            "cost": round(self.cost(), 4),
            "input_tokens": tokens_in,
            "cached_pct": round(100 * cached / tokens_in) if tokens_in else 0,
            "output_tokens": sum(e["output"] for e in self.llm.values()),
            "calls": sum(e["calls"] for e in self.llm.values()),
            "audio_minutes": round(sum(self.audio_seconds.values()) / 60, 1),
            "searches": self.searches,
        }
