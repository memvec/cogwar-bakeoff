"""Per-provider token pricing + running-cost tracking, shared by every pass
that can spend real money (extract_entities.py, detect_stance.py,
run_full_pipeline.py).

Pricing verified 2026-08 against each provider's own pricing reference
(ai.google.dev/gemini-api/docs/pricing for Gemini Flash;
claude-api skill / claude.com pricing for Claude Sonnet 4.6). USD per
1,000,000 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING = {
    "anthropic": {"input": 3.00, "output": 15.00},  # claude-sonnet-4-6
    "gemini": {"input": 0.75, "output": 3.75},  # gemini flash, introductory rate through 2026-12-31
}


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING[provider]
    return input_tokens / 1_000_000 * rates["input"] + output_tokens / 1_000_000 * rates["output"]


@dataclass
class CostTracker:
    """Accumulates spend across a run and enforces an optional hard cap.

    `add()` returns False (and does not add) when adding this call's cost
    would push the running total past `max_cost` -- the caller is
    responsible for stopping cleanly at that point (this class never raises
    or aborts anything itself, so a partial run's already-committed DB
    writes are never at risk).
    """

    provider: str
    max_cost: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    total_cost: float = 0.0

    def would_exceed(self, extra_cost: float = 0.0) -> bool:
        if self.max_cost is None:
            return False
        return (self.total_cost + extra_cost) > self.max_cost

    def add(self, input_tokens: int, output_tokens: int) -> float:
        """Record one call's usage. Returns the cost of this call. Does NOT
        check the cap -- call would_exceed() beforehand if you need to
        decide whether to make the call at all; this just records what
        already happened."""
        call_cost = estimate_cost(self.provider, input_tokens, output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.total_cost += call_cost
        return call_cost

    def progress_line(self, done: int, total: int) -> str:
        cap_str = f" (cap ${self.max_cost:.2f})" if self.max_cost is not None else ""
        return (
            f"[{self.provider}] {done}/{total} done -- {self.calls} calls, "
            f"running cost ${self.total_cost:.4f}{cap_str}"
        )
