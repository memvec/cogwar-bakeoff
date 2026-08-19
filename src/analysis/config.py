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
