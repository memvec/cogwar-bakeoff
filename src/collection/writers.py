"""Shared output writers for collected data. Used by every source collector
(src/collection/telegram/, and future src/collection/youtube/,
src/collection/news/) — no source-specific code belongs here.

Layout under the configured raw data root (config.DATA_OUTPUT_PATH, default
data/raw/):
    items/<run_id>.jsonl        one JSON object per line, one file per run
    edges/<run_id>.parquet      one file per run
    observations/<run_id>.parquet   one file per run
    payloads/<run_id>/<ref>.json     raw source payloads, one file per object

Every writer here is append-by-construction: it writes exactly one new,
uniquely-named file per collection run and refuses (`FileExistsError`) to
reopen a run_id that already has output, so re-running the collector always
adds to history instead of overwriting it. This matters most for
observations — doc §9, "Author/channel nodes — observation history +
reputation fields" — but the same guarantee is applied uniformly to items
and edges too, since a partial re-run silently clobbering a prior run's
items/edges would be just as bad for reproducibility (doc §5).
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from collection.schema import Edge, Item, Observation


def _guard_not_exists(out_path: Path, run_id: str) -> None:
    if out_path.exists():
        raise FileExistsError(
            f"Output for run_id={run_id!r} already exists at {out_path}. "
            "Each collection run must use a unique run_id; refusing to overwrite."
        )


def write_items(items: list[Item], run_id: str, output_dir: Path) -> Path:
    """Write this run's items (channel/author nodes + content items) as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.jsonl"
    _guard_not_exists(out_path, run_id)

    with out_path.open("w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    return out_path


def write_edges(edges: list[Edge], run_id: str, output_dir: Path) -> Path:
    """Write this run's edges as parquet.

    `evidence` varies in shape across edge types (forward vs. reply vs.
    mention each carry different keys), which would give pyarrow's struct
    inference an inconsistent schema across rows. We serialize `evidence` to
    a JSON string column instead — a stable parquet schema regardless of
    edge_type mix, queryable in DuckDB via its json functions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.parquet"
    _guard_not_exists(out_path, run_id)

    rows = []
    for edge in edges:
        row = edge.model_dump(mode="json")
        row["evidence"] = json.dumps(row["evidence"])
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return out_path


def write_observations(
    observations: list[Observation], run_id: str, output_dir: Path
) -> Path:
    """Append this run's author/channel observations as a new parquet file.

    Writes exactly one new file, named after `run_id`, under `output_dir`
    (default data/raw/observations/). Re-running the collector — which must
    mint a fresh run_id per run (doc §5, `collection_run_id`) — therefore
    always adds a file; it never opens or rewrites a prior run's file.

    Downstream, DuckDB reads the whole directory as one time-series table:
    read_parquet('<output_dir>/*.parquet').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.parquet"
    _guard_not_exists(out_path, run_id)

    table = pa.Table.from_pylist([o.model_dump(mode="json") for o in observations])
    pq.write_table(table, out_path)
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
