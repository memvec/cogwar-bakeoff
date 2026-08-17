"""Collection layer settings and secrets.

Loads Telegram API credentials and collection settings from the environment
(via a local .env file, see .env.example) and fails loudly at import time if
required secrets are missing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH")


def _fail_missing(missing: list[str]) -> None:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(missing)
        + ".\n"
        "Create a .env file in the project root (see .env.example) with "
        "your Telegram API credentials from https://my.telegram.org, "
        "then try again."
    )


_missing = [name for name in _REQUIRED_VARS if not os.getenv(name)]
if _missing:
    _fail_missing(_missing)

_raw_api_id = os.environ["TELEGRAM_API_ID"]
try:
    TELEGRAM_API_ID: int = int(_raw_api_id)
except ValueError as exc:
    raise RuntimeError(
        f"TELEGRAM_API_ID must be an integer, got {_raw_api_id!r}. "
        "Check your .env file against .env.example."
    ) from exc

TELEGRAM_API_HASH: str = os.environ["TELEGRAM_API_HASH"]
TELEGRAM_SESSION_NAME: str = os.getenv("TELEGRAM_SESSION_NAME", "cogwar_collect")

# Only needed for the one-time interactive Telethon login (client.start());
# not required at import time since most code paths (e.g. schema/writers
# tests) never touch the network. Set it in .env before running the
# collector for the first time.
TELEGRAM_PHONE: str | None = os.getenv("TELEGRAM_PHONE")

DATA_OUTPUT_PATH: Path = Path(os.getenv("DATA_OUTPUT_PATH", "data/raw/"))
ITEMS_DIR: Path = DATA_OUTPUT_PATH / "items"
EDGES_DIR: Path = DATA_OUTPUT_PATH / "edges"
OBSERVATIONS_DIR: Path = DATA_OUTPUT_PATH / "observations"
PAYLOADS_DIR: Path = DATA_OUTPUT_PATH / "payloads"
