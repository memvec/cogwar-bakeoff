"""Analysis layer settings and secrets.

Deliberately independent of collection.config: analysis only needs the
Anthropic API key, not Telegram/YouTube credentials. Same reasoning as
processing.config being independent of collection.config -- importing
analysis.config must not force unrelated secrets to be configured.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = ("ANTHROPIC_API_KEY",)


def _fail_missing(missing: list[str]) -> None:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(missing)
        + ".\n"
        "Create a .env file in the project root (see .env.example) with "
        "your Anthropic API key from https://console.anthropic.com/settings/keys, "
        "then try again."
    )


_missing = [name for name in _REQUIRED_VARS if not os.getenv(name)]
if _missing:
    _fail_missing(_missing)

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

# The processed DB this pass reads items/edges from (src/processing/config.py's
# DB_PATH) -- referenced by path, not by importing processing.config, to keep
# this module's only dependency the Anthropic key.
PROCESSED_DB_PATH = Path("data/processed/cogwar.duckdb")

ANALYSIS_DATA_PATH = Path("data/analysis")
ANALYSIS_DB_PATH = ANALYSIS_DATA_PATH / "analysis.duckdb"

SEED_ENTITIES_PATH = Path("configs/seed_entities.json")

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Gemini is an alternative provider behind the same swappable interfaces
# (entities.EntityExtractor / stance.StanceDetector) -- optional at import
# time, unlike ANTHROPIC_API_KEY above, because anthropic remains the
# default provider and nothing that only uses it should be forced to
# configure Gemini too. get_gemini_api_key() below is where this fails
# loud, the moment something actually tries to select "gemini".
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Gemini Flash model id. Both 3.6-flash and 3.7-flash hit this project's
# generate_requests_per_model_per_day quota (10,000/day) on 2026-08-22 --
# quotas are tracked per MODEL, so heavy usage on one doesn't affect the
# other's bucket, but a big enough corpus run saturates each in turn.
# 3.6-flash's bucket had the most time to age out as of 2026-08-23 05:5x IST
# (confirmed via a 25-call burst test, 0 failures) while 3.7-flash was still
# fully exhausted at the same check -- so 3.6-flash is the active choice
# again for now. Keep checking both if this one saturates too (verified
# against ai.google.dev/gemini-api/docs/pricing as of 2026-08).
GEMINI_MODEL = "gemini-3.6-flash"


def get_gemini_api_key() -> str:
    """Fail loud only when Gemini is actually selected as a provider.

    IMPORTANT: use a PAID-TIER Gemini API key. Google's free tier trains on
    submitted data by default -- unacceptable here, since this pipeline's
    content is real (if public) account text. Enable billing on the
    AI Studio / Cloud project the key comes from before using it.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing required environment variable: GEMINI_API_KEY.\n"
            "Create a .env entry with a PAID-TIER Gemini API key from "
            "https://aistudio.google.com/apikey -- confirm billing is enabled on "
            "that project first; a free-tier key trains on submitted data, which "
            "is not acceptable for this pipeline's content.\n"
            "Then try again."
        )
    return GEMINI_API_KEY
