"""Shared output writers for collected data. Used by every source collector
(src/collection/telegram/, and future src/collection/youtube/,
src/collection/news/) — no source-specific code belongs here.

Layout under the configured raw data root (config.DATA_OUTPUT_PATH, default
data/raw/):
    items/<key>.jsonl        one JSON object per line, one file per key
    edges/<key>.parquet      one file per key
    observations/<key>.parquet   one file per key
    payloads/<run_id>/<ref>.json     raw source payloads, one file per object

`<key>` defaults to `run_id` (the whole run as one file, the original
behavior) but callers may pass a more specific `file_key` — e.g. one file
per channel per run — so a logical unit smaller than "the whole run" can be
durably written on its own. This is what lets a caller enforce write before
checkpoint: advancing a checkpoint is only safe once *that unit's* data is
confirmed on disk, not "eventually, when the whole run finishes" (see
telegram/collector.py's collect_and_persist_all).

Every writer here is append-by-construction: it writes exactly one new,
uniquely-named file per key and refuses (`FileExistsError`) to reopen a key
that already has output, so re-running the collector always adds to history
instead of overwriting it. This matters most for observations — doc §9,
"Author/channel nodes — observation history + reputation fields" — but the
same guarantee is applied uniformly to items and edges too, since a partial
re-run silently clobbering a prior run's items/edges would be just as bad
for reproducibility (doc §5).

Every write here is fsync'd before the function returns: the caller's next
action (e.g. advancing a checkpoint) can safely assume the data is durable
on disk, not just handed to a buffer that a killed process might lose.
"""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from collection.schema import Edge, Item, Observation


def _guard_not_exists(out_path: Path, key: str) -> None:
    if out_path.exists():
        raise FileExistsError(
            f"Output for key={key!r} already exists at {out_path}. "
            "Each collection run (or sub-unit, e.g. channel) must use a "
            "unique key; refusing to overwrite."
        )


def _fsync_path(path: Path) -> None:
    """fsync a file that's already been fully written and closed (e.g. by pyarrow's own internal I/O, which we don't hold a handle to)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_items(
    items: list[Item], run_id: str, output_dir: Path, *, file_key: str | None = None
) -> Path:
    """Write items (channel/author nodes + content items) as JSONL. `file_key` overrides the filename (defaults to `run_id`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key = file_key or run_id
    out_path = output_dir / f"{key}.jsonl"
    _guard_not_exists(out_path, key)

    with out_path.open("w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    return out_path


def write_edges(
    edges: list[Edge], run_id: str, output_dir: Path, *, file_key: str | None = None
) -> Path:
    """Write edges as parquet. `file_key` overrides the filename (defaults to `run_id`).

    `evidence` varies in shape across edge types (forward vs. reply vs.
    mention each carry different keys), which would give pyarrow's struct
    inference an inconsistent schema across rows. We serialize `evidence` to
    a JSON string column instead — a stable parquet schema regardless of
    edge_type mix, queryable in DuckDB via its json functions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key = file_key or run_id
    out_path = output_dir / f"{key}.parquet"
    _guard_not_exists(out_path, key)

    rows = []
    for edge in edges:
        row = edge.model_dump(mode="json")
        row["evidence"] = json.dumps(row["evidence"])
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    _fsync_path(out_path)
    return out_path


def write_observations(
    observations: list[Observation],
    run_id: str,
    output_dir: Path,
    *,
    file_key: str | None = None,
) -> Path:
    """Append author/channel observations as a new parquet file. `file_key` overrides the filename (defaults to `run_id`).

    Writes exactly one new file under `output_dir` (default
    data/raw/observations/); never opens or rewrites an existing one.

    Downstream, DuckDB reads the whole directory as one time-series table:
    read_parquet('<output_dir>/*.parquet').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key = file_key or run_id
    out_path = output_dir / f"{key}.parquet"
    _guard_not_exists(out_path, key)

    table = pa.Table.from_pylist([o.model_dump(mode="json") for o in observations])
    pq.write_table(table, out_path)
    _fsync_path(out_path)
    return out_path


def write_raw_payload(payload: dict, ref_name: str, run_id: str, output_dir: Path) -> str:
    """Write one object's raw source payload as JSON; return the path to store as its Item/Provenance `raw_payload_ref`.

    `ref_name` must uniquely identify the object within this run (e.g.
    f"channel_{channel_id}" or f"message_{channel_id}_{message_id}"); files
    land under output_dir/<run_id>/, so different runs never collide.
    """
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{ref_name}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, default=str)
    return str(out_path)
