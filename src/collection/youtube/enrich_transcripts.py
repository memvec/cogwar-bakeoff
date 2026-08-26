"""YouTube transcript enrichment pass -- CLI entrypoint.

    uv run python -m collection.youtube.enrich_transcripts --limit 10

Separate from collection on purpose: fetching a transcript is a much
heavier, much more rate-limit-sensitive operation than pulling
search/videos.list metadata, and it's something you'd want to re-run
independently (e.g. once a better transcript provider exists) without
re-collecting anything. Loads already-collected youtube_video items from
every file under data/raw/items/, fetches a transcript for each one still
missing `source_specific.transcript` (via the provider below), and rewrites
the affected files in place with the transcript block added -- everything
else about the item (text, engagement, entities, ...) is untouched.

Swapping providers (e.g. to a future Whisper-based one) is a one-line
change -- see `provider = ...` in main() below.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from collection import config, enrich
from collection.schema import Item, SourceType
from collection.youtube.transcript import (
    TranscriptProvider,
    TranscriptResult,
    YoutubeTranscriptApiProvider,
)

# youtube-transcript-api scrapes YouTube's own timedtext endpoints and will
# get an IP flagged/blocked if hammered -- this is not a JSON API with a
# generous quota like videos.list. Default kept as the fallback for direct
# run_enrichment() callers (e.g. tests); main() always passes --sleep explicitly.
REQUEST_SLEEP_SECONDS = 1.5


class EnrichStats:
    def __init__(self) -> None:
        self.processed = 0
        self.with_transcript = 0
        self.failed_quality = 0
        self.without_by_reason: dict[str, int] = {}
        self.language_breakdown: dict[str, int] = {}
        self.script_breakdown: dict[str, int] = {}

    def record_failure(self, reason: str | None) -> None:
        key = reason or "unknown"
        self.without_by_reason[key] = self.without_by_reason.get(key, 0) + 1


def _load_items(path: Path) -> list[Item]:
    with path.open() as f:
        return [Item.model_validate(json.loads(line)) for line in f if line.strip()]


def _rewrite_items_file(path: Path, items: list[Item]) -> None:
    """Overwrite an existing items file in place with updated items.

    Deliberately NOT writers.write_items(): that function guards against
    overwriting a run's output because collection runs must never clobber
    each other's history (doc §9's append-only rule). This pass has the
    opposite intent -- it's backfilling a field onto already-collected
    items, not producing a new run's output -- so overwriting the same file
    is exactly what should happen here.

    Atomic (temp file + rename), same discipline as checkpoints.py's
    _save_store -- run_enrichment calls this after every single transcribed
    video (see its docstring), not just once per file, so a crash mid-write
    must never leave a corrupt/truncated items file and lose already-earned
    transcription work.
    """
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    tmp_path.replace(path)


def build_transcript_block(result: TranscriptResult, fetched_at: datetime) -> dict:
    """Map a TranscriptResult into the source_specific.transcript block (doc §1's per-source pattern).

    Runs script/language detection on the transcript text specifically --
    reusing enrich.py, doc §9's shared enrichment -- since spoken content
    can be a different language/script than the title (e.g. an
    auto-generated English transcript on a Hindi-titled video, or vice
    versa). Stored under transcript_script/transcript_language so this
    never collides with or overwrites the item's own top-level
    script/language_detected, which describe the title+description text.
    """
    normalized = enrich.normalize_text(result.text)
    script = enrich.detect_script(result.text)
    return {
        "text": result.text,
        "language_code": result.language_code,
        "is_generated": result.is_generated,
        "provider_name": result.provider_name,
        "fetched_at": fetched_at.isoformat(),
        "text_hash": enrich.compute_text_hash(normalized),
        "transcript_script": script,
        "transcript_language": enrich.detect_language(result.text, script=script),
    }


def run_enrichment(
    items_dir: Path,
    provider: TranscriptProvider,
    limit: int,
    sleep_seconds: float = REQUEST_SLEEP_SECONDS,
    target_video_ids: set[str] | None = None,
    quality_check: Callable[[TranscriptResult], tuple[bool, str | None]] | None = None,
    on_attempt: Callable[[], None] | None = None,
) -> EnrichStats:
    """Per-video checkpointing: the affected file is rewritten after EVERY
    successfully transcribed video (not batched to once per file) -- a
    provider like WhisperTranscriptProvider can spend real wall-clock time
    per video, so a crash/interrupt partway through a file must not lose
    videos that were already transcribed earlier in that same file. Since
    "already has source_specific.transcript" is also the skip condition
    below, this doubles as the resume mechanism: re-running after an
    interruption just picks up wherever it left off, no separate state file
    needed.

    target_video_ids, when given, restricts processing to exactly that set
    (e.g. a scoped subset of videos to transcribe) -- every other
    youtube_video item is left untouched, same as if it had already been
    processed. None (the default) processes every youtube_video item, the
    original behavior.

    quality_check, when given, runs on every successful fetch() result
    BEFORE it's trusted and stored (e.g. transcript.detect_whisper_
    hallucination) -- (is_bad, reason). A bad result is never written as
    source_specific.transcript; instead the item is marked
    source_specific.transcript_failed_quality = True (also checked by the
    skip condition above, so it's never retried forever) and the failure
    is counted in stats.failed_quality rather than stats.with_transcript.

    on_attempt, when given, is called once per video actually attempted
    (regardless of outcome) -- lets a caller track its own progress/ETA
    without this function needing to know the caller's notion of "total".
    """
    stats = EnrichStats()
    # Only a youtube-run file can ever contain a youtube_video item --
    # writers.py names every raw items file after its collector run_id, and
    # youtube runs are always named "youtube_<timestamp>_<hex>"
    # (collector.py's main()). items_dir is shared with Telegram's raw
    # output too, which is by far the larger share of files/items at this
    # corpus's scale (1260 telegram files vs 7 youtube files, ~1.85M items
    # total) -- globbing and pydantic-parsing every Telegram file on every
    # enrichment run wastes minutes of CPU for nothing, since none of them
    # can ever match, and at this scale that's long enough to look like a hang.
    file_paths = sorted(items_dir.glob("youtube_*.jsonl"))

    for path in file_paths:
        if stats.processed >= limit:
            break
        items = _load_items(path)

        for item in items:
            if stats.processed >= limit:
                break
            if item.source_type != SourceType.youtube_video:
                continue
            if "transcript" in item.source_specific:
                continue  # already enriched by a previous run -- don't re-hammer the API
            if item.source_specific.get("transcript_failed_quality"):
                continue  # already tried and failed the quality guard -- don't retry forever

            video_id = item.source_specific.get("video_id") or item.source_native_id
            if target_video_ids is not None and video_id not in target_video_ids:
                continue

            result = provider.fetch(video_id)
            stats.processed += 1
            if on_attempt is not None:
                on_attempt()

            if result is None:
                stats.record_failure(provider.last_failure_reason)
                time.sleep(sleep_seconds)
                continue

            if quality_check is not None:
                is_bad, reason = quality_check(result)
                if is_bad:
                    item.source_specific["transcript_failed_quality"] = True
                    _rewrite_items_file(path, items)
                    stats.failed_quality += 1
                    print(f"[enrich_transcripts] {video_id}: FAILED quality guard -- {reason}", flush=True)
                    time.sleep(sleep_seconds)
                    continue

            fetched_at = datetime.now(UTC)
            block = build_transcript_block(result, fetched_at)
            item.source_specific["transcript"] = block
            _rewrite_items_file(path, items)

            stats.with_transcript += 1
            lang = block["transcript_language"] or "unknown"
            script = block["transcript_script"] or "unknown"
            stats.language_breakdown[lang] = stats.language_breakdown.get(lang, 0) + 1
            stats.script_breakdown[script] = stats.script_breakdown.get(script, 0) + 1

            print(
                f"[enrich_transcripts] {video_id}: transcript_language={block['transcript_language']!r} "
                f"transcript_script={block['transcript_script']!r} is_generated={result.is_generated}",
                flush=True,
            )
            time.sleep(sleep_seconds)

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube transcript enrichment (docs/collection_schema.md)")
    parser.add_argument("--limit", type=int, default=20, help="Max videos to process")
    parser.add_argument(
        "--sleep",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help="Seconds to sleep between transcript fetches (raise this if IP-flagged)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Swap boundary: change this one line to use a different TranscriptProvider later.
    provider = YoutubeTranscriptApiProvider()

    stats = run_enrichment(config.ITEMS_DIR, provider, args.limit, sleep_seconds=args.sleep)

    print("\n--- Transcript enrichment summary ---")
    print(f"Videos processed: {stats.processed}")
    print(f"With transcripts: {stats.with_transcript}")
    print(f"Failed quality guard: {stats.failed_quality}")
    print("Without transcripts, by reason:")
    for reason, count in sorted(stats.without_by_reason.items()):
        print(f"  {reason}: {count}")
    if not stats.without_by_reason:
        print("  (none)")
    print("Language breakdown (of videos with transcripts):")
    for lang, count in sorted(stats.language_breakdown.items()):
        print(f"  {lang}: {count}")
    print("Script breakdown (of videos with transcripts):")
    for script, count in sorted(stats.script_breakdown.items()):
        print(f"  {script}: {count}")


if __name__ == "__main__":
    main()
