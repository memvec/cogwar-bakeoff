"""Regression test for the checkpoint-durability invariant: a channel's
checkpoint must never advance until that channel's items/edges/observations
are durably on disk.

Simulates an interrupted run (collect_and_persist_all processes channel A
fully, then channel B raises mid-processing, standing in for the process
being killed) and asserts A is both written and checkpointed while B is
neither -- so a subsequent run treats B as never having been touched
(clean backfill, no silent gap) rather than skipping content that was
never actually persisted.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from collection import checkpoints, config
from collection.schema import (
    Edge,
    EdgeOrigin,
    EdgeType,
    FetchMethod,
    Item,
    Observation,
    Provenance,
    SourceType,
)
from collection.telegram import collector as telegram_collector


def _make_channel_result(channel_key: str, run_id: str, high_water_mark: str) -> telegram_collector.ChannelResult:
    """A real, valid ChannelResult -- same shape collect_channel would return for a channel with one message."""
    collected_at = datetime.now(UTC)
    provenance = Provenance(
        collector_id="telegram_v1",
        collected_at=collected_at,
        fetch_method=FetchMethod.api,
        collection_run_id=run_id,
        raw_payload_ref=f"data/raw/payloads/{run_id}/channel_{channel_key}.json",
    )
    channel_item = Item(
        source_type=SourceType.channel,
        source_native_id=channel_key,
        author_native_id=channel_key,
        author_display_name=f"Test Channel {channel_key}",
        raw_payload_ref=provenance.raw_payload_ref,
        provenance=provenance,
    )
    message_item = Item(
        source_type=SourceType.telegram,
        source_native_id=f"{channel_key}:{high_water_mark}",
        parent_item_id=channel_item.item_id,
        text="test message",
        raw_payload_ref=provenance.raw_payload_ref,
        provenance=provenance,
    )
    edge = Edge(
        edge_type=EdgeType.mention,
        src_item_id=message_item.item_id,
        dst_external_ref="@someone",
        observed_at=collected_at,
        origin=EdgeOrigin.collected,
        evidence={"mention_text": "@someone"},
    )
    observation = Observation(
        node_item_id=channel_item.item_id,
        observed_at=collected_at,
        subscriber_or_follower_count=100,
        post_count_seen=1,
        collection_run_id=run_id,
    )
    return telegram_collector.ChannelResult(
        channel_key=channel_key,
        channel_item=channel_item,
        observation=observation,
        message_items=[message_item],
        edges=[edge],
        new_high_water_mark=high_water_mark,
        mode="backfill",
    )


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect config's output dirs and the checkpoint store to a tmp_path -- no real data/raw or data/state touched."""
    items_dir = tmp_path / "items"
    edges_dir = tmp_path / "edges"
    observations_dir = tmp_path / "observations"
    monkeypatch.setattr(config, "ITEMS_DIR", items_dir)
    monkeypatch.setattr(config, "EDGES_DIR", edges_dir)
    monkeypatch.setattr(config, "OBSERVATIONS_DIR", observations_dir)
    monkeypatch.setattr(checkpoints, "STORE_PATH", tmp_path / "checkpoints.json")
    return tmp_path


def test_interrupted_run_leaves_unwritten_channel_unckeckpointed(
    isolated_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "test-run-interrupted"

    async def fake_collect_channel(client, handle, run_id, max_per_channel, months, stats):
        if handle == "channel_a":
            return _make_channel_result("channel_a_key", run_id, high_water_mark="100")
        if handle == "channel_b":
            # Simulate the process being killed while channel B is in flight --
            # an uncaught, unexpected termination, not a handled dead-channel skip.
            raise RuntimeError("simulated crash mid-channel-B")
        raise AssertionError(f"unexpected handle {handle!r}")

    monkeypatch.setattr(telegram_collector, "collect_channel", fake_collect_channel)

    stats = telegram_collector.RunStats()
    with pytest.raises(RuntimeError, match="simulated crash mid-channel-B"):
        asyncio.run(
            telegram_collector.collect_and_persist_all(
                client=None,
                handles=["channel_a", "channel_b"],
                run_id=run_id,
                max_per_channel=10,
                months=1,
                stats=stats,
            )
        )

    # --- Channel A: fully processed before the crash -- must be both written and checkpointed ---
    a_items = list(config.ITEMS_DIR.glob(f"{run_id}__channel_a_key.jsonl"))
    a_edges = list(config.EDGES_DIR.glob(f"{run_id}__channel_a_key.parquet"))
    a_observations = list(config.OBSERVATIONS_DIR.glob(f"{run_id}__channel_a_key.parquet"))
    assert len(a_items) == 1, "channel A's items file must be durably written"
    assert len(a_edges) == 1, "channel A's edges file must be durably written"
    assert len(a_observations) == 1, "channel A's observations file must be durably written"
    assert a_items[0].read_text().strip() != "", "channel A's items file must have content"

    a_checkpoint = checkpoints.get_checkpoint("telegram", "channel_a_key")
    assert a_checkpoint is not None, "channel A's checkpoint must have advanced"
    assert a_checkpoint.high_water_mark == "100"
    assert a_checkpoint.last_run_id == run_id

    # --- Channel B: crashed mid-processing -- must be neither written nor checkpointed ---
    b_items = list(config.ITEMS_DIR.glob(f"{run_id}__channel_b_key.jsonl"))
    b_edges = list(config.EDGES_DIR.glob(f"{run_id}__channel_b_key.parquet"))
    b_observations = list(config.OBSERVATIONS_DIR.glob(f"{run_id}__channel_b_key.parquet"))
    assert b_items == [], "channel B must have no items file -- it never finished collecting"
    assert b_edges == [], "channel B must have no edges file"
    assert b_observations == [], "channel B must have no observations file"

    b_checkpoint = checkpoints.get_checkpoint("telegram", "channel_b_key")
    assert b_checkpoint is None, (
        "channel B's checkpoint must NOT have advanced -- with no checkpoint, "
        "the next run will treat it as never-collected and backfill cleanly, "
        "rather than silently skipping content that was never persisted"
    )

    # Only channel A's completion was recorded in stats -- B never reached the increment.
    assert stats.channels == 1
