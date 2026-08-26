"""Whisper-based (offline, local) transcript enrichment -- CLI entrypoint.

    uv run python -m collection.youtube.transcribe_whisper --sample-size 10
    uv run python -m collection.youtube.transcribe_whisper --full --limit 10000

Targets videos with no scraped captions available at all: videos from the
Pakistani news channels (collection/youtube's channel-based collection
pass, data/seeds/anti_india_youtube_channels_jul24_aug24_2026.csv) plus any
other video matching a topic keyword on title/description
(configs/topic_keywords.json), already-transcribed videos excluded.

Two-phase by design (--sample-size vs --full): a Whisper pass costs real
wall-clock compute per video (audio download + local ASR), unlike the
scrape-based YoutubeTranscriptApiProvider's cheap HTTP call, so a small
timed sample run is how you size the full run's cost before committing to
it. Reuses collection.youtube.enrich_transcripts.run_enrichment (same
source_specific.transcript shape, same per-item durability/checkpointing,
same idempotent "already has a transcript -> skip" resume behavior) with
WhisperTranscriptProvider swapped in and a target_video_ids filter applied
-- no new persistence mechanism, just the existing enrichment pass pointed
at a scoped set and a different provider.

Target-set identification is a deliberate, modest exception to the
collection layer's usual independence from the analysis layer: "on-topic"
is fundamentally an analysis-layer scoping concept (analysis/scope.py's
keyword list + matching logic), and duplicating that rule here would just
be two copies going out of sync. analysis.scope has no dependency on
analysis.config (no ANTHROPIC_API_KEY / provider coupling) -- importing it
costs nothing beyond duckdb, which this module needs anyway to query the
processed DB for the target set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import duckdb

from analysis.scope import load_topic_keywords, matched_keywords_for_text
from collection import config, enrich
from collection.checkpoints import STORE_PATH as CHECKPOINT_STORE_PATH
from collection.youtube.enrich_transcripts import EnrichStats, run_enrichment
from collection.youtube.transcript import (
    TranscriptProvider,
    TranscriptResult,
    WhisperTranscriptProvider,
    detect_whisper_hallucination,
)

PROCESSED_DB_PATH = Path("data/processed/cogwar.duckdb")
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_MODEL_SIZE = "small"
# Some target videos are multi-hour livestream re-uploads (confirmed: up to
# ~11.9 hours in this corpus, 22 of 1985 candidates exceed 1 hour) --
# whisper's memory/CPU cost scales with audio length with no built-in cap,
# so an unbounded video can turn a "10-video timed sample" into an
# unbounded one and blow past available memory. Excluded from the target
# set (reported separately, not silently dropped) rather than transcribed;
# revisit with a chunked-transcription approach if these turn out to matter.
DEFAULT_MAX_DURATION_SECONDS = 3600


def load_pakistani_channel_ids() -> list[str]:
    """Channel ids collected in the YouTube channel-based pass targeting the
    20 Pakistani news channels (data/seeds/anti_india_youtube_channels_
    jul24_aug24_2026.csv, collector.py's --channels-csv). Read from
    checkpoints.json's 'youtube_channel' namespace rather than re-resolving
    the CSV -- that would cost real search.list quota for no reason, since
    we already collected against it once. 16, not 20: 2 of the 20 CSV rows
    resolved to a channel another row already covered (see the resolved-
    mapping report from that collection run), and 2 (GNN, City 42) had zero
    recent uploads when collected, so they never got a checkpoint -- neither
    omission changes the target video set, since a channel with 0 collected
    videos contributes 0 target videos regardless.
    """
    with CHECKPOINT_STORE_PATH.open() as f:
        store = json.load(f)
    return sorted(store.get("youtube_channel", {}).keys())


def identify_target_videos(
    con: duckdb.DuckDBPyConnection, max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS
) -> tuple[list[dict], list[dict]]:
    """Videos from the Pakistani channels, plus any other video matching a
    topic keyword on title/description -- deduplicated (a single query
    naturally dedupes by item_id), already-transcribed videos excluded.
    Returns (targets, excluded_long) -- the second list is videos that
    would otherwise qualify but exceed max_duration_seconds (see
    DEFAULT_MAX_DURATION_SECONDS's docstring), kept separate so the caller
    can report them rather than silently dropping them.

    Matches against `text` only (title+description, already concatenated
    at collection time in youtube/mapping.py) -- NOT transcript text, since
    these are exactly the videos that don't have one yet.

    Uses author_native_id (not source_specific->>'channel_id') for the
    channel match: mapping.py's build_video_item sets author_native_id to
    the video's channelId directly, so it's already a properly-typed
    VARCHAR column -- avoids a real DuckDB quirk where filtering on
    source_type before a JSON-extracted field on this heterogeneous
    (Telegram + YouTube) source_specific column raises a spurious numeric
    cast error, apparently from type inference colliding across the two
    source types' differently-shaped JSON.
    """
    channel_ids = set(load_pakistani_channel_ids())
    keyword_pairs = load_topic_keywords()

    rows = con.execute(
        """
        SELECT item_id, source_specific->>'video_id' AS video_id, author_native_id AS channel_id,
               author_display_name, text, (source_specific->>'duration_seconds')::BIGINT AS duration_seconds
        FROM items
        WHERE source_type = 'youtube_video'
          AND (source_specific -> 'transcript' ->> 'text') IS NULL
        """
    ).fetchall()

    targets = []
    excluded_long = []
    for item_id, video_id, channel_id, author_display_name, text, duration_seconds in rows:
        is_pakistani_channel = channel_id in channel_ids
        matches = matched_keywords_for_text(text or "", keyword_pairs)
        is_on_topic = bool(matches)
        if not is_pakistani_channel and not is_on_topic:
            continue
        if is_pakistani_channel and is_on_topic:
            source = "channel+keyword"
        elif is_pakistani_channel:
            source = "channel"
        else:
            source = "keyword"
        record = {
            "item_id": item_id,
            "video_id": video_id,
            "channel_id": channel_id,
            "author_display_name": author_display_name,
            "text": text,
            "source": source,
            "duration_seconds": duration_seconds,
            "matched_keywords": sorted({kw for _, kw in matches}),
        }
        if duration_seconds is not None and duration_seconds > max_duration_seconds:
            excluded_long.append(record)
        else:
            targets.append(record)
    return targets, excluded_long


def print_target_report(targets: list[dict], excluded_long: list[dict]) -> None:
    from_channel_only = sum(1 for t in targets if t["source"] == "channel")
    from_keyword_only = sum(1 for t in targets if t["source"] == "keyword")
    both = sum(1 for t in targets if t["source"] == "channel+keyword")
    print(f"[transcribe_whisper] target videos (not yet transcribed): {len(targets)}")
    print(f"  Pakistani channel only: {from_channel_only}")
    print(f"  keyword match only (other channels): {from_keyword_only}")
    print(f"  both: {both}")
    if excluded_long:
        total_hours = sum(t["duration_seconds"] for t in excluded_long) / 3600
        print(
            f"  excluded for length (> {DEFAULT_MAX_DURATION_SECONDS // 60} min): {len(excluded_long)} videos, "
            f"{total_hours:.1f} total hours -- not in target set, handle separately if needed"
        )


# Exactly the two focus topics from configs/topic_keywords.json for this
# scoped run -- deliberately narrower than identify_target_videos' full
# 5-topic sweep (drops Indus Waters Treaty, the broad Kashmir bucket, and
# the generic India-Pakistan-conflict bucket) and, unlike that function,
# requires BOTH conditions rather than either: Pakistani channel AND a
# focus-topic keyword match. Pahalgam's keywords live under "Operation
# Sindoor" here (see topic_keywords.json) since the request bundles it with
# the Sindoor narrative, even though it's also (separately) filed under
# "Kashmir" for the broader keyword list.
FOCUS_TOPICS = {"Operation Sindoor", "Turkey-Saudi-Pakistan pact"}

# If the focus-scoped count comes back in the thousands rather than a few
# hundred, that's a sign the keyword match is catching more than intended
# -- worth a warning rather than silently transcribing all of it.
FOCUS_TARGET_WARNING_THRESHOLD = 1000


def identify_focus_scoped_targets(
    con: duckdb.DuckDBPyConnection,
    focus_topics: set[str] = FOCUS_TOPICS,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
) -> tuple[list[dict], list[dict]]:
    """Pakistani-channel videos matching one of `focus_topics`' keywords on
    title/description -- the intersection, not the union identify_target_videos
    computes. Also excludes videos already marked transcript_failed_quality
    (the hallucination guard's persistent "don't retry forever" marker) in
    addition to already-transcribed ones -- both are resumability
    conditions for this pass. Returns (targets, excluded_long), same shape
    as identify_target_videos.
    """
    channel_ids = set(load_pakistani_channel_ids())
    all_keyword_pairs = load_topic_keywords()
    focus_keyword_pairs = [(t, kw) for t, kw in all_keyword_pairs if t in focus_topics]
    if not focus_keyword_pairs:
        raise ValueError(
            f"No keywords found for focus topics {focus_topics!r} -- "
            "check the topic names against configs/topic_keywords.json"
        )

    rows = con.execute(
        """
        SELECT item_id, source_specific->>'video_id' AS video_id, author_native_id AS channel_id,
               author_display_name, text, (source_specific->>'duration_seconds')::BIGINT AS duration_seconds
        FROM items
        WHERE source_type = 'youtube_video'
          AND (source_specific -> 'transcript' ->> 'text') IS NULL
          AND (source_specific ->> 'transcript_failed_quality') IS NULL
        """
    ).fetchall()

    targets = []
    excluded_long = []
    for item_id, video_id, channel_id, author_display_name, text, duration_seconds in rows:
        if channel_id not in channel_ids:
            continue
        matches = matched_keywords_for_text(text or "", focus_keyword_pairs)
        if not matches:
            continue
        record = {
            "item_id": item_id,
            "video_id": video_id,
            "channel_id": channel_id,
            "author_display_name": author_display_name,
            "text": text,
            "duration_seconds": duration_seconds,
            "matched_keywords": sorted({kw for _, kw in matches}),
            "matched_topics": sorted({t for t, _ in matches}),
        }
        if duration_seconds is not None and duration_seconds > max_duration_seconds:
            excluded_long.append(record)
        else:
            targets.append(record)
    return targets, excluded_long


def print_focus_target_report(targets: list[dict], excluded_long: list[dict]) -> None:
    print(f"[transcribe_whisper] focus-scoped target videos: {len(targets)}")
    if excluded_long:
        total_hours = sum(t["duration_seconds"] for t in excluded_long) / 3600
        print(
            f"  excluded for length (> {DEFAULT_MAX_DURATION_SECONDS // 60} min): {len(excluded_long)} videos, "
            f"{total_hours:.1f} total hours"
        )
    by_topic: dict[str, int] = {}
    for t in targets:
        for topic in t["matched_topics"]:
            by_topic[topic] = by_topic.get(topic, 0) + 1
    for topic, count in sorted(by_topic.items()):
        print(f"  matched {topic!r}: {count}")
    if len(targets) > FOCUS_TARGET_WARNING_THRESHOLD:
        print(
            f"  WARNING: {len(targets)} is much larger than the expected few hundred -- "
            "the focus-topic keyword match may be too broad. Check configs/topic_keywords.json "
            "before proceeding with the full run.",
            flush=True,
        )


def pick_sample(targets: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Prefer a mix: roughly half from the Pakistani channels (the most
    reliable Urdu/Hindi signal in this target set), half from keyword-only
    matches -- so the timed sample and quality check reflect both
    populations, per the request for "ideally some Hindi/Urdu."
    """
    rng = random.Random(seed)
    channel_targets = [t for t in targets if t["source"] in ("channel", "channel+keyword")]
    keyword_only_targets = [t for t in targets if t["source"] == "keyword"]
    rng.shuffle(channel_targets)
    rng.shuffle(keyword_only_targets)

    half = sample_size // 2
    sample = channel_targets[:half] + keyword_only_targets[: sample_size - half]
    if len(sample) < sample_size:
        leftovers = channel_targets[half:] + keyword_only_targets[sample_size - half :]
        sample += leftovers[: sample_size - len(sample)]
    return sample[:sample_size]


class _TimingWrapper(TranscriptProvider):
    """Records per-call timing + the raw TranscriptResult while delegating
    to a real WhisperTranscriptProvider -- used only for the timed-sample
    report below; the full run uses the provider directly, no wrapper.
    """

    def __init__(self, inner: WhisperTranscriptProvider) -> None:
        self._inner = inner
        self.records: list[dict] = []

    def fetch(self, video_id: str) -> TranscriptResult | None:
        result = self._inner.fetch(video_id)
        self.last_failure_reason = self._inner.last_failure_reason
        timing = dict(self._inner.last_timing_seconds or {})
        self.records.append(
            {
                "video_id": video_id,
                "success": result is not None,
                "text": result.text if result else None,
                "language_code": result.language_code if result else None,
                "failure_reason": self._inner.last_failure_reason,
                **timing,
            }
        )
        return result


def run_timed_sample(
    targets: list[dict], sample_size: int, model_size: str, seed: int
) -> tuple[list[dict], EnrichStats, EnrichStats]:
    """Transcribes `sample_size` videos from the target set, then re-runs the
    exact same target set once more to demonstrate resumability (everything
    should be skipped the second time, since it now has a transcript).
    Returns (per-video timing/text records, first-pass stats, resume-pass stats).
    """
    sample = pick_sample(targets, sample_size, seed)
    sample_video_ids = {t["video_id"] for t in sample}
    by_video_id = {t["video_id"]: t for t in sample}

    print(f"\n[transcribe_whisper] loading faster-whisper model {model_size!r} ...", flush=True)
    provider = WhisperTranscriptProvider(model_size=model_size)
    wrapper = _TimingWrapper(provider)

    print(f"[transcribe_whisper] transcribing {len(sample)} sample videos ...", flush=True)
    overall_start = time.monotonic()
    first_pass_stats = run_enrichment(
        config.ITEMS_DIR, wrapper, limit=len(sample), target_video_ids=sample_video_ids
    )
    overall_elapsed = time.monotonic() - overall_start

    print("\n--- Per-video timing ---")
    total_download = 0.0
    total_transcribe = 0.0
    for r in wrapper.records:
        meta = by_video_id.get(r["video_id"], {})
        dl = r.get("download_seconds", 0.0) or 0.0
        tr = r.get("transcribe_seconds", 0.0) or 0.0
        total_download += dl
        total_transcribe += tr
        status = "OK" if r["success"] else f"FAILED ({r['failure_reason']})"
        print(
            f"  {r['video_id']} [{meta.get('source', '?')}] {meta.get('author_display_name', '?')!r}: "
            f"download={dl:.1f}s transcribe={tr:.1f}s -- {status}"
        )

    n = max(len(wrapper.records), 1)
    per_video_avg = overall_elapsed / n
    print(
        f"\nTotal for {len(wrapper.records)} videos: {overall_elapsed:.1f}s wall-clock "
        f"(download={total_download:.1f}s, transcribe={total_transcribe:.1f}s)"
    )
    print(f"Average per video: {per_video_avg:.1f}s")

    n_targets = len(targets)
    extrapolated_seconds = per_video_avg * n_targets
    print(
        f"Extrapolated for full target set ({n_targets} videos): "
        f"{extrapolated_seconds / 3600:.2f} hours ({extrapolated_seconds / 60:.1f} minutes)"
    )

    print("\n[transcribe_whisper] verifying resumability -- re-running the same target set ...", flush=True)
    resume_stats = run_enrichment(
        config.ITEMS_DIR, provider, limit=len(sample), target_video_ids=sample_video_ids
    )
    print(
        f"  Re-run processed: {resume_stats.processed} "
        f"(expected 0 -- every sample video now already has a transcript)"
    )

    return wrapper.records, first_pass_stats, resume_stats


def print_transcript_samples(records: list[dict], n: int = 2) -> None:
    successful = [r for r in records if r["success"] and r["text"]]
    print(f"\n--- Transcript samples ({min(n, len(successful))} of {len(successful)} successful) ---")
    # Prefer showing a language mix if we have one.
    non_english = [r for r in successful if (r.get("language_code") or "").lower() not in ("en", "en-us", "")]
    chosen = (non_english[:1] + [r for r in successful if r not in non_english[:1]])[:n]
    for r in chosen:
        script = enrich.detect_script(r["text"])
        language = enrich.detect_language(r["text"], script=script)
        print(f"\n  video {r['video_id']} -- language_code={r['language_code']!r} detected={language}/{script}")
        print(f"  {r['text'][:600]!r}")


def _apply_niceness(nice_value: int) -> None:
    """Lowers this process's OS scheduling priority so a long throttled
    run doesn't compete with foreground work on the same machine. Best-
    effort: os.nice() ADDS to the current niceness (POSIX semantics, not an
    absolute set) and can fail (e.g. sandboxed/restricted environments) --
    neither case should abort the run, just proceed at default priority.
    """
    try:
        new_niceness = os.nice(nice_value)
        print(f"[transcribe_whisper] set process niceness to {new_niceness} (lower priority)", flush=True)
    except OSError as e:
        print(f"[transcribe_whisper] could not set niceness (continuing at default priority): {e}", flush=True)


class _ProgressTracker:
    """N/total, elapsed, ETA -- printed after every attempted video (on_attempt
    callback into run_enrichment), so a long throttled run's progress is visible
    without needing run_enrichment itself to know the caller's notion of "total".
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.attempted = 0
        self.start = time.monotonic()

    def tick(self) -> None:
        self.attempted += 1
        elapsed = time.monotonic() - self.start
        avg = elapsed / self.attempted
        eta_seconds = avg * (self.total - self.attempted)
        print(
            f"[transcribe_whisper] progress {self.attempted}/{self.total} -- "
            f"elapsed={elapsed / 60:.1f}min ETA={eta_seconds / 60:.1f}min",
            flush=True,
        )


def run_focus_scoped(
    targets: list[dict],
    model_size: str,
    cpu_threads: int,
    nice_value: int,
    limit: int,
) -> None:
    """The throttled, quality-guarded, resumable full run over a focus-scoped
    target set. Reuses run_enrichment exactly as the sample path does --
    same per-video durability/checkpointing, same source_specific.transcript
    shape -- with detect_whisper_hallucination wired in as quality_check and
    a progress-printing on_attempt callback.
    """
    _apply_niceness(nice_value)
    print(
        f"[transcribe_whisper] cpu_threads={cpu_threads} (of {os.cpu_count()} total cores)",
        flush=True,
    )

    print(f"[transcribe_whisper] loading faster-whisper model {model_size!r} ...", flush=True)
    provider = WhisperTranscriptProvider(model_size=model_size, cpu_threads=cpu_threads)

    target_video_ids = {t["video_id"] for t in targets}
    tracker = _ProgressTracker(total=len(target_video_ids))

    print(f"[transcribe_whisper] transcribing {len(target_video_ids)} focus-scoped videos ...", flush=True)
    overall_start = time.monotonic()
    stats = run_enrichment(
        config.ITEMS_DIR,
        provider,
        limit=limit,
        target_video_ids=target_video_ids,
        quality_check=detect_whisper_hallucination,
        on_attempt=tracker.tick,
    )
    overall_elapsed = time.monotonic() - overall_start

    n_unavailable = sum(stats.without_by_reason.values())
    print("\n--- Focus-scoped run complete ---")
    print(f"Transcribed OK: {stats.with_transcript}")
    print(f"Failed quality guard: {stats.failed_quality}")
    print(f"Unavailable/blocked (provider failure): {n_unavailable}")
    for reason, count in sorted(stats.without_by_reason.items()):
        print(f"  {reason}: {count}")
    print("Language breakdown (of videos with transcripts):")
    for lang, count in sorted(stats.language_breakdown.items()):
        print(f"  {lang}: {count}")
    print(f"Total wall-clock: {overall_elapsed / 3600:.2f} hours ({overall_elapsed / 60:.1f} minutes)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper-based YouTube transcript enrichment")
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Timed sample size (default 10)"
    )
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, help="faster-whisper model size (default small)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the sample draw")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the FULL (broad) target set instead of the timed sample -- only after reviewing the sample report",
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help=(
            "Run the focus-scoped (Pakistani channels AND Sindoor/Pahalgam/Makkah-pact keywords), "
            "quality-guarded, throttled full run"
        ),
    )
    parser.add_argument("--limit", type=int, default=100_000, help="Max videos to process in --full/--focus mode")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="faster-whisper CPU thread cap for --focus (default: roughly half of os.cpu_count())",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=10,
        help="OS niceness increment applied before a --focus run (default 10; higher = lower priority)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    con = duckdb.connect(str(PROCESSED_DB_PATH), read_only=True)

    if args.focus:
        print("[transcribe_whisper] identifying focus-scoped target set ...", flush=True)
        targets, excluded_long = identify_focus_scoped_targets(con)
        con.close()
        print_focus_target_report(targets, excluded_long)

        cpu_threads = args.cpu_threads if args.cpu_threads is not None else max(1, (os.cpu_count() or 2) // 2)
        run_focus_scoped(targets, args.model_size, cpu_threads, args.nice, args.limit)
        return

    print("[transcribe_whisper] identifying target set ...", flush=True)
    targets, excluded_long = identify_target_videos(con)
    con.close()
    print_target_report(targets, excluded_long)

    if not args.full:
        records, _first_pass_stats, _resume_stats = run_timed_sample(
            targets, args.sample_size, args.model_size, args.seed
        )
        print_transcript_samples(records, n=2)
        print(
            "\n[transcribe_whisper] SAMPLE COMPLETE -- not running the full target set. "
            "Re-invoke with --full once the timing/quality above looks good.",
            flush=True,
        )
        return

    print(f"\n[transcribe_whisper] FULL RUN -- {len(targets)} target videos, limit={args.limit}", flush=True)
    provider = WhisperTranscriptProvider(model_size=args.model_size)
    target_video_ids = {t["video_id"] for t in targets}
    stats = run_enrichment(config.ITEMS_DIR, provider, limit=args.limit, target_video_ids=target_video_ids)

    print("\n--- Whisper enrichment summary ---")
    print(f"Videos processed: {stats.processed}")
    print(f"With transcripts: {stats.with_transcript}")
    print("Without transcripts, by reason:")
    for reason, count in sorted(stats.without_by_reason.items()):
        print(f"  {reason}: {count}")
    if not stats.without_by_reason:
        print("  (none)")
    print("Language breakdown (of videos with transcripts):")
    for lang, count in sorted(stats.language_breakdown.items()):
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
