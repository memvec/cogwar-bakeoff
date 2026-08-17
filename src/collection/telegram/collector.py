"""Telegram collector -- CLI entrypoint.

    uv run python -m collection.telegram.collector --limit 50

Authenticates via Telethon (config.py credentials + TELEGRAM_SESSION_NAME
session file), pulls channel + message history for a set of seed handles,
maps everything into canonical Items/Edges/Observations (telegram/mapping.py),
and writes it via the shared writers (collection.writers). Politely: small
sleeps between channels, respects FloodWaitError's mandated wait, and skips
dead/invalid/private channels without crashing the run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.functions.channels import GetFullChannelRequest

from collection import config, writers
from collection.schema import Edge, Item, Observation
from collection.telegram import mapping

# Small default test set -- edit directly, or pass --handles / --seeds-csv.
DEFAULT_TEST_HANDLES = [
    "@doamuslims",
    "@News_Pakistan",
    "@DDGeopolitics",
    "@Slavyangrad",
]

CHANNEL_SLEEP_SECONDS = 2.0
MESSAGE_BATCH_SLEEP_SECONDS = 0.5
MESSAGE_BATCH_SIZE = 20


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
    handle: str,
    run_id: str,
    limit: int,
    since: datetime | None,
    stats: RunStats,
) -> tuple[Item, Observation, list[Item], list[Edge]] | None:
    """Collect one channel's identity, reputation observation, and message history.

    Returns None (recording the skip in `stats`) for any dead, invalid, or
    private channel, or a flood-wait too long to be worth blocking the whole
    run for -- one bad handle must never crash the run.
    """
    try:
        entity = await _call_with_flood_wait(client.get_entity, handle)
        full_response = await _call_with_flood_wait(client, GetFullChannelRequest(entity))
        full = full_response.full_chat
        collected_at = datetime.now(UTC)

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

        iter_kwargs: dict = {"limit": limit}
        if since is not None:
            iter_kwargs["offset_date"] = since
            iter_kwargs["reverse"] = True

        message_count = 0
        async for message in client.iter_messages(entity, **iter_kwargs):
            message_count += 1
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

        observation = mapping.build_channel_observation(
            entity, full, channel_item.item_id, message_count, run_id, collected_at
        )
        return channel_item, observation, message_items, edges

    except FloodWaitError as e:
        print(
            f"[collector] skipping {handle}: FloodWait of {e.seconds}s -- too long to block "
            "this run for; a future run will pick it back up.",
            flush=True,
        )
        stats.dead_channels.append(handle)
        return None
    except (RPCError, ValueError, TypeError) as e:
        print(f"[collector] skipping {handle}: {type(e).__name__}: {e}", flush=True)
        stats.dead_channels.append(handle)
        return None


async def run_collection(
    handles: list[str], run_id: str, limit: int, since: datetime | None
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
            result = await collect_channel(client, handle, run_id, limit, since, stats)
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
    parser.add_argument("--limit", type=int, default=50, help="Max messages per channel")
    parser.add_argument("--handles", nargs="+", default=None, help="Explicit list of @handles")
    parser.add_argument(
        "--seeds-csv", type=Path, default=None, help="CSV with a 'Telegram handle' column"
    )
    parser.add_argument("--since", type=str, default=None, help="ISO date cutoff, e.g. 2026-01-01")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handles = load_seed_handles(args.handles, args.seeds_csv)
    since = (
        datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    )
    run_id = f"telegram_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"

    print(f"[collector] run_id={run_id} handles={handles} limit={args.limit}", flush=True)

    items, edges, observations, stats = asyncio.run(
        run_collection(handles, run_id, args.limit, since)
    )

    if items:
        writers.write_items(items, run_id, config.ITEMS_DIR)
    if edges:
        writers.write_edges(edges, run_id, config.EDGES_DIR)
    if observations:
        writers.write_observations(observations, run_id, config.OBSERVATIONS_DIR)

    print("\n--- Collection summary ---")
    print(f"Channels collected: {stats.channels}")
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
