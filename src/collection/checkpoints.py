"""Checkpoint store for incremental collection. Shared by every source
collector -- Telegram, YouTube, and future sources all read/write through
this same module and file.

Persists, per (source, key), a high_water_mark plus last_run_at/last_run_id
-- the boundary marking what's already been collected. An absent checkpoint
means "never collected this key before" -> backfill mode; a present one
means incremental mode (fetch only what's newer than high_water_mark).

`source` is the collector name ("telegram", "youtube", ...). `key` is a
source-specific identity for the thing being tracked -- a Telegram channel's
numeric id (as a string, stable across handle changes), a YouTube search
query string, etc. `high_water_mark` is source-specific too: Telegram uses
the last collected message_id; YouTube uses the last published_at ISO
string seen for that query.

Plain JSON, not DuckDB: checkpoints are simple, low-volume key-value state
that collectors need to read/write with no query language, and collection
must not depend on the processing layer's duckdb dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

STORE_PATH = Path("data/state/checkpoints.json")


@dataclass
class Checkpoint:
    high_water_mark: str
    last_run_at: str
    last_run_id: str


def _load_store() -> dict:
    if not STORE_PATH.exists():
        return {}
    with STORE_PATH.open() as f:
        return json.load(f)


def _save_store(store: dict) -> None:
    """Write via a temp file + atomic rename so a crash mid-write can never leave a corrupt/partial checkpoints.json."""
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STORE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(store, f, indent=2, sort_keys=True)
    tmp_path.replace(STORE_PATH)


def get_checkpoint(source: str, key: str) -> Checkpoint | None:
    """None means this (source, key) has never been collected -- caller should backfill."""
    raw = _load_store().get(source, {}).get(key)
    if raw is None:
        return None
    return Checkpoint(**raw)


def set_checkpoint(source: str, key: str, high_water_mark: str, run_id: str) -> None:
    store = _load_store()
    store.setdefault(source, {})[key] = asdict(
        Checkpoint(
            high_water_mark=str(high_water_mark),
            last_run_at=datetime.now(UTC).isoformat(),
            last_run_id=run_id,
        )
    )
    _save_store(store)
