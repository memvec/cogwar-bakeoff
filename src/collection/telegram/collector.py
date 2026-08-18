"""Telegram collector -- CLI entrypoint.

    uv run python -m collection.telegram.collector --max-per-channel 50

Authenticates via Telethon (config.py credentials + TELEGRAM_SESSION_NAME
session file), pulls channel + message history for a set of seed handles,
maps everything into canonical Items/Edges/Observations (telegram/mapping.py),
and writes it via the shared writers (collection.writers). Politely: small
sleeps between channels, respects FloodWaitError's mandated wait, and skips
dead/invalid/private channels without crashing the run.

Incremental via checkpoints.py: a channel with no prior checkpoint backfills
--months back; a channel with a checkpoint only fetches messages newer than
its high_water_mark (last collected message_id). Observations are exempt --
collected every run regardless of mode, since the follower-count time
series needs a data point on every pass, not just on new content.

Before collecting, mines already-collected edges for forward-origin
channels we've never pulled ourselves and adds them to this run's seed set
(see mine_forward_target_channel_ids) -- newly-added channels have no
checkpoint, so they naturally backfill on this run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import uuid
from calendar import monthrange
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.functions.channels import GetFullChannelRequest

from collection import checkpoints, config, writers
from collection.schema import Edge, Item, Observation
from collection.telegram import mapping

# Small default test set -- edit directly, or pass --handles / --seeds-csv.
DEFAULT_TEST_HANDLES = [
    "@doamuslims",
    "@News_Pakistan",
    "@DDGeopolitics",
    "@Slavyangrad",
]

CHECKPOINT_SOURCE = "telegram"
DEFAULT_BACKFILL_MONTHS = 4

CHANNEL_SLEEP_SECONDS = 2.0
MESSAGE_BATCH_SLEEP_SECONDS = 0.5
MESSAGE_BATCH_SIZE = 20

# Forward edges to an uncollected origin channel serialize as
# dst_external_ref = "telegram:<channel_id>:<msg_id>" (telegram/mapping.py
# build_forward_edge); this is the inverse parse.
_FWD_EXTERNAL_RE = re.compile(r"^telegram:(-?\d+):(-?\d+)$")


def load_seed_handles(handles_arg: list[str] | None, seeds_csv: Path | None) -> list[str]:
    """Explicit --handles wins; else --seeds-csv's 'Telegram handle' column; else the default test set."""
    if handles_arg:
        return handles_arg
    if seeds_csv is not None:
        with seeds_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            col = next(
                (c for c in reader.fieldnames or [] if c.strip().lower() == "telegram handle"),
                None,
            )
            if col is None:
                raise ValueError(
                    f"{seeds_csv} has no 'Telegram handle' column. Found: {reader.fieldnames}"
                )
            return [row[col].strip() for row in reader if row[col].strip()]
    return DEFAULT_TEST_HANDLES


def mine_forward_target_channel_ids(edges_dir: Path) -> list[int]:
    """Scan raw edges/*.parquet for forward-origin channels we've never collected.

    A forward edge whose origin channel we haven't collected resolves to
    dst_item_id=NULL, dst_external_ref="telegram:<channel_id>:<msg_id>".
    Extract the distinct channel_ids, excluding any we already have a
    checkpoint for (checkpoints.py's definition of "already collected") --
    what's left is new seed material: channels other collected channels
    forwarded from, that we've never pulled ourselves. Resolution isn't
    guaranteed (Telegram channel entities need an access_hash Telethon may
    not have cached) -- collect_channel already skips unresolvable handles
    gracefully, so failures here just fall through as normal dead channels.
    """
    candidates: set[int] = set()
    for path in sorted(edges_dir.glob("*.parquet")):
        table = pq.read_table(str(path), columns=["edge_type", "dst_item_id", "dst_external_ref"])
        for row in table.to_pylist():
            if row["edge_type"] != "forward" or row["dst_item_id"] is not None:
                continue
            ref = row.get("dst_external_ref")
            m = _FWD_EXTERNAL_RE.match(ref) if ref else None
            if m:
                candidates.add(int(m.group(1)))
    return sorted(
        cid for cid in candidates if checkpoints.get_checkpoint(CHECKPOINT_SOURCE, str(cid)) is None
    )


def _months_ago(months: int, from_dt: datetime | None = None) -> datetime:
    """Calendar-correct `months` before `from_dt` (default now, UTC). No dateutil dependency."""
    dt = from_dt or datetime.now(UTC)
    total_months = dt.year * 12 + (dt.month - 1) - months
    year, month0 = divmod(total_months, 12)
    month = month0 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _fifo_reader(prompt_label: str):
    """Returns a callable Telethon can use for phone/code/password prompts.

    Normally this just calls input(), i.e. the expected behavior when a
    person runs this collector themselves in a real terminal. If
    TELEGRAM_LOGIN_FIFO is set, it instead blocks reading a line from that
    FIFO -- lets a non-interactive caller relay the value through a pipe
    instead of a real TTY.
    """

    def _read() -> str:
        fifo_path = os.environ.get("TELEGRAM_LOGIN_FIFO")
        if not fifo_path:
            return input(f"Please enter your {prompt_label}: ")
        print(f"[collector] waiting for {prompt_label} via FIFO {fifo_path} ...", flush=True)
        with open(fifo_path) as f:
            return f.readline().strip()

    return _read


class RunStats:
    def __init__(self) -> None:
        self.channels = 0
        self.backfill_channels = 0
        self.incremental_channels = 0
        self.items = 0
        self.edges_by_type: dict[str, int] = {}
        self.observations = 0
        self.dead_channels: list[str] = []

    def record_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            self.edges_by_type[edge.edge_type.value] = (
                self.edges_by_type.get(edge.edge_type.value, 0) + 1
            )


async def _call_with_flood_wait(fn, *args, **kwargs):
    """Call an async Telethon function, respecting one FloodWaitError by sleeping the mandated duration and retrying once."""
    try:
        return await fn(*args, **kwargs)
    except FloodWaitError as e:
        print(f"[collector] FloodWait: sleeping {e.seconds}s as instructed by Telegram...", flush=True)
        await asyncio.sleep(e.seconds)
        return await fn(*args, **kwargs)


async def collect_channel(
    client: TelegramClient,
    handle: str | int,
    run_id: str,
    max_per_channel: int,
    months: int,
    stats: RunStats,
) -> tuple[Item, Observation, list[Item], list[Edge]] | None:
    """Collect one channel's identity, reputation observation, and message history.

    Checkpoint-driven: no prior checkpoint for this channel -> backfill the
    last `months`; a checkpoint present -> incremental, only messages newer
    than its high_water_mark. Both modes iterate oldest-to-newest
    (reverse=True) so that if there's more new content than
    `max_per_channel` allows, the checkpoint still only advances to the
    newest message actually fetched -- never skipping a gap that a future
    run wouldn't come back for.

    Returns None (recording the skip in `stats`) for any dead, invalid, or
    private channel, or a flood-wait too long to be worth blocking the whole
    run for -- one bad handle must never crash the run.
    """
    try:
        entity = await _call_with_flood_wait(client.get_entity, handle)
        full_response = await _call_with_flood_wait(client, GetFullChannelRequest(entity))
        full = full_response.full_chat
        collected_at = datetime.now(UTC)

        channel_key = str(entity.id)
        checkpoint = checkpoints.get_checkpoint(CHECKPOINT_SOURCE, channel_key)

        # Confidence isn't threaded further -- account_created_at stays None
        # for Telegram regardless (platform limitation); see mapping.py.
        created_at, _created_at_confidence = await mapping.approximate_channel_created_at(
            client, entity
        )

        channel_payload_ref = writers.write_raw_payload(
            {"entity": entity.to_dict(), "full_chat": full.to_dict()},
            ref_name=f"channel_{entity.id}",
            run_id=run_id,
            output_dir=config.PAYLOADS_DIR,
        )
        channel_item = mapping.build_channel_item(
            entity, full, created_at, run_id, channel_payload_ref, collected_at
        )

        item_lookup: mapping.ItemLookup = {}
        message_items: list[Item] = []
        edges: list[Edge] = []

        if checkpoint is None:
            mode = "backfill"
            iter_kwargs: dict = {
                "limit": max_per_channel,
                "offset_date": _months_ago(months),
                "reverse": True,
            }
        else:
            mode = "incremental"
            iter_kwargs = {
                "limit": max_per_channel,
                "min_id": int(checkpoint.high_water_mark),
                "reverse": True,
            }

        message_count = 0
        max_msg_id_seen: int | None = None
        async for message in client.iter_messages(entity, **iter_kwargs):
            message_count += 1
            max_msg_id_seen = message.id if max_msg_id_seen is None else max(max_msg_id_seen, message.id)
            msg_payload_ref = writers.write_raw_payload(
                message.to_dict(),
                ref_name=f"message_{entity.id}_{message.id}",
                run_id=run_id,
                output_dir=config.PAYLOADS_DIR,
            )
            item = mapping.build_message_item(
                message, entity, channel_item.item_id, run_id, msg_payload_ref, collected_at
            )
            message_items.append(item)
            item_lookup[(entity.id, message.id)] = item.item_id

            fwd_edge = mapping.build_forward_edge(message, item, item_lookup, collected_at)
            if fwd_edge:
                edges.append(fwd_edge)
            reply_edge = mapping.build_reply_edge(message, item, entity, item_lookup, collected_at)
            if reply_edge:
                edges.append(reply_edge)
            edges.extend(mapping.build_mention_edges(item, item.entities, collected_at))

            if message_count % MESSAGE_BATCH_SIZE == 0:
                await asyncio.sleep(MESSAGE_BATCH_SLEEP_SECONDS)

        # Observations are exempt from incremental filtering -- always a
        # fresh snapshot, every run (doc §9 time series).
        observation = mapping.build_channel_observation(
            entity, full, channel_item.item_id, message_count, run_id, collected_at
        )

        new_hwm = max_msg_id_seen if max_msg_id_seen is not None else (
            checkpoint.high_water_mark if checkpoint else None
        )
        if new_hwm is not None:
            checkpoints.set_checkpoint(CHECKPOINT_SOURCE, channel_key, str(new_hwm), run_id)

        print(f"[collector] {handle}: {mode} -- {message_count} new item(s)", flush=True)
        if mode == "backfill":
            stats.backfill_channels += 1
        else:
            stats.incremental_channels += 1

        return channel_item, observation, message_items, edges

    except FloodWaitError as e:
        print(
            f"[collector] skipping {handle}: FloodWait of {e.seconds}s -- too long to block "
            "this run for; a future run will pick it back up.",
            flush=True,
        )
        stats.dead_channels.append(str(handle))
        return None
    except (RPCError, ValueError, TypeError) as e:
        print(f"[collector] skipping {handle}: {type(e).__name__}: {e}", flush=True)
        stats.dead_channels.append(str(handle))
        return None


async def run_collection(
    handles: list[str | int], run_id: str, max_per_channel: int, months: int
) -> tuple[list[Item], list[Edge], list[Observation], RunStats]:
    stats = RunStats()
    all_items: list[Item] = []
    all_edges: list[Edge] = []
    all_observations: list[Observation] = []

    client = TelegramClient(
        config.TELEGRAM_SESSION_NAME, config.TELEGRAM_API_ID, config.TELEGRAM_API_HASH
    )
    await client.start(
        phone=config.TELEGRAM_PHONE or _fifo_reader("phone number"),
        code_callback=_fifo_reader("login code"),
        password=_fifo_reader("2FA password"),
    )
    try:
        for handle in handles:
            result = await collect_channel(client, handle, run_id, max_per_channel, months, stats)
            if result is None:
                continue
            channel_item, observation, message_items, edges = result
            stats.channels += 1
            stats.items += 1 + len(message_items)
            stats.observations += 1
            stats.record_edges(edges)

            all_items.append(channel_item)
            all_items.extend(message_items)
            all_edges.extend(edges)
            all_observations.append(observation)

            await asyncio.sleep(CHANNEL_SLEEP_SECONDS)
    finally:
        await client.disconnect()

    return all_items, all_edges, all_observations, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram collector (docs/collection_schema.md)")
    parser.add_argument(
        "--max-per-channel", type=int, default=50, help="Max messages per channel per run"
    )
    parser.add_argument(
        "--months", type=int, default=DEFAULT_BACKFILL_MONTHS, help="Backfill window in months (new channels only)"
    )
    parser.add_argument("--handles", nargs="+", default=None, help="Explicit list of @handles")
    parser.add_argument(
        "--seeds-csv", type=Path, default=None, help="CSV with a 'Telegram handle' column"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_handles = load_seed_handles(args.handles, args.seeds_csv)

    mined = mine_forward_target_channel_ids(config.EDGES_DIR)
    print(f"[collector] forward-mined {len(mined)} new candidate channel(s) from existing edges", flush=True)

    handles: list[str | int] = [*base_handles, *mined]
    run_id = f"telegram_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"

    print(
        f"[collector] run_id={run_id} handles={handles} "
        f"max_per_channel={args.max_per_channel} months={args.months}",
        flush=True,
    )

    items, edges, observations, stats = asyncio.run(
        run_collection(handles, run_id, args.max_per_channel, args.months)
    )

    if items:
        writers.write_items(items, run_id, config.ITEMS_DIR)
    if edges:
        writers.write_edges(edges, run_id, config.EDGES_DIR)
    if observations:
        writers.write_observations(observations, run_id, config.OBSERVATIONS_DIR)

    print("\n--- Collection summary ---")
    print(f"Channels collected: {stats.channels} (backfill={stats.backfill_channels}, incremental={stats.incremental_channels})")
    print(f"Items written: {stats.items}")
    print("Edges by type:")
    for edge_type, count in sorted(stats.edges_by_type.items()):
        print(f"  {edge_type}: {count}")
    if not stats.edges_by_type:
        print("  (none)")
    print(f"Observations: {stats.observations}")
    dead = stats.dead_channels
    print(f"Dead channels skipped: {len(dead)}" + (f" ({', '.join(dead)})" if dead else ""))


if __name__ == "__main__":
    main()
