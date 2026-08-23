"""Shared retry/backoff wrapper for Gemini API calls.

Used by both entities.GeminiEntityExtractor and stance.GeminiStanceDetector
-- consolidated here rather than duplicated in each, because the
free-tier-vs-transient distinction below is exactly the kind of logic that
must not silently drift between two copies when real spend is on the line
(a full-corpus run is a paid-tier, real-money operation).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")


class GeminiFreeTierError(RuntimeError):
    """Raised when Gemini's own error response indicates the configured key
    is on the free tier. FATAL, never retried: free-tier requests train on
    submitted data (see analysis.config.get_gemini_api_key's docstring), and
    this pipeline's content (real, if public, account text) must not be
    exposed to that. The run should stop immediately, not silently keep
    burning through a 20-requests/day quota while sending more text.
    """


class GeminiBillingExhaustedError(RuntimeError):
    """Raised when Gemini's own error response indicates the account's
    prepaid balance is depleted (distinct from a transient rate limit, and
    distinct from the free-tier case above) -- observed in practice as a 429
    RESOURCE_EXHAUSTED whose message says prepayment credits are depleted.
    FATAL, never retried: this will not resolve itself on a timer the way a
    per-minute rate limit does, so retrying it for several minutes (as a
    generic 429 would be) just wastes wall-clock time before failing anyway.
    Requires topping up billing at https://ai.studio/projects before any
    further call can succeed.
    """


class GeminiDailyQuotaExhaustedError(RuntimeError):
    """Raised when Gemini's own error response indicates a per-model daily
    request quota (generate_requests_per_model_per_day) has been used up --
    observed in practice at 10,000 requests/day for a single model on a
    single project. Distinct from a transient per-minute rate limit (a 503
    or a plain 429) and from GeminiBillingExhaustedError (this has nothing
    to do with account balance) -- it is quota is per MODEL, so switching to
    a different model id on the same project has its own separate bucket.
    FATAL, never retried: the response's own retryDelay is typically hours,
    far longer than any reasonable backoff, so retrying wastes wall-clock
    time before failing anyway. Resolves automatically when the quota
    resets (~24h rolling window) -- the retry delay Gemini reports is
    included in the message when parseable.
    """


def _extract_retry_delay_seconds(message: str) -> int | None:
    match = _RETRY_DELAY_RE.search(message)
    return int(match.group(1)) if match else None


def call_with_retry(fn: Callable[[], T], attempts: int = 6, base_backoff_seconds: float = 4.0) -> T:
    """Call `fn` (a zero-arg thunk wrapping one generate_content call).

    - A 429 whose message names the free tier -> GeminiFreeTierError,
      immediately, no retry (see class docstring).
    - A 429 whose message names depleted prepayment/billing credits ->
      GeminiBillingExhaustedError, immediately, no retry (see class
      docstring) -- an account-balance problem, not a rate limit, so
      retrying it cannot help.
    - A 429 whose message names an exhausted per-day quota ->
      GeminiDailyQuotaExhaustedError, immediately, no retry (see class
      docstring) -- the reset window is hours, so retrying cannot help
      within this process's lifetime.
    - Any other 429 (a genuine paid-tier rate limit -- transient, real, just
      less severe) or a 503 (transient server overload, observed repeatedly
      against newly-launched Flash models in practice) -> retry with
      exponential backoff up to `attempts` tries.
    - Anything else (bad key, malformed request, etc.) propagates on the
      first attempt -- this is not a general-purpose safety net.
    """
    from google.genai import errors

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except errors.ClientError as e:
            message = str(e)
            lower = message.lower()
            if "free_tier" in lower or "freetier" in lower:
                raise GeminiFreeTierError(
                    "Gemini API returned a free-tier quota error -- the configured "
                    "GEMINI_API_KEY is NOT on the paid tier. Stopping immediately: "
                    "free-tier requests train on submitted data, which this pipeline's "
                    "content must not be exposed to. Enable billing on the key's project "
                    "(or swap in a paid-tier key) before retrying.\n"
                    f"Original error: {message}"
                ) from e
            if "prepayment" in lower or "prepaid" in lower or "credits are depleted" in lower:
                raise GeminiBillingExhaustedError(
                    "Gemini API reports the account's prepaid credits are depleted. "
                    "Stopping immediately rather than retrying (this is a billing-balance "
                    "problem, not a transient rate limit -- it will not resolve on its own). "
                    "Top up billing at https://ai.studio/projects, then rerun the same "
                    "command -- everything already processed is cached and will be skipped.\n"
                    f"Original error: {message}"
                ) from e
            if "per_day" in lower or "perday" in lower:
                retry_seconds = _extract_retry_delay_seconds(message)
                retry_note = f" Gemini reports a retry delay of ~{retry_seconds}s (~{retry_seconds / 3600:.1f}h)." if retry_seconds else ""
                raise GeminiDailyQuotaExhaustedError(
                    "Gemini API reports the per-model daily request quota is exhausted for "
                    "this model on this project. Stopping immediately rather than retrying "
                    "(the reset window is hours, not seconds, so retrying wastes time before "
                    "failing anyway)." + retry_note + " Options: wait for the quota to reset, "
                    "request a quota increase in the Cloud console, or switch GEMINI_MODEL to a "
                    "different model id (daily quotas are tracked per model, so a different "
                    "model has its own separate bucket).\n"
                    f"Original error: {message}"
                ) from e
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(base_backoff_seconds * (2**attempt))
                    continue
            raise
        except errors.ServerError as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(base_backoff_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error
