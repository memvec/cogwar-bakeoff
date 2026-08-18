"""Transcript provider interface for YouTube videos.

TranscriptProvider is the swap boundary: today's implementation
(YoutubeTranscriptApiProvider) scrapes existing captions via the
youtube-transcript-api library. A future Whisper-based or commercial-API
provider becomes just another implementation of this same interface --
nothing in enrich_transcripts.py or youtube/mapping.py needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float


@dataclass
class TranscriptResult:
    text: str  # full transcript, segments joined with a single space
    language_code: str
    is_generated: bool
    provider_name: str
    segments: list[TranscriptSegment] | None = None


class TranscriptProvider(ABC):
    """Swap boundary for transcript sourcing.

    Implementations must never raise -- a missing/unavailable transcript is
    a normal, expected outcome (most videos have no manual captions, many
    have no captions at all), not an error condition. fetch() returns None
    rather than propagating exceptions, so callers never need
    provider-specific try/except blocks.
    """

    #: Best-effort diagnostic, set by fetch() just before it returns None
    #: (e.g. "no_captions_in_preferred_languages" / "disabled" /
    #: "video_unavailable" / "blocked"). NOT part of the required contract
    #: -- a provider may leave this None always and still satisfy the
    #: interface. Purely a convenience for callers that want a "why" for
    #: their own logging/summary; a future provider (Whisper, a commercial
    #: API) is free to use different reason strings, or none at all.
    last_failure_reason: str | None = None

    @abstractmethod
    def fetch(self, video_id: str) -> TranscriptResult | None:
        """Fetch a transcript for one video. None if unavailable for any reason."""
        raise NotImplementedError


# Preferred transcript languages, in priority order -- Hindi/Urdu first
# since this project's primary interest is Hindi/English/Urdu CIB content;
# falls back to English, then gives up rather than grabbing an unrelated
# language's captions.
DEFAULT_LANGUAGE_PRIORITY = ["hi", "ur", "en"]


class YoutubeTranscriptApiProvider(TranscriptProvider):
    """Scrapes existing (manual or auto-generated) YouTube captions via
    youtube-transcript-api. No audio processing -- only works for videos
    that already have captions in one of `language_priority`.
    """

    def __init__(self, language_priority: list[str] | None = None) -> None:
        from youtube_transcript_api import YouTubeTranscriptApi

        self._api = YouTubeTranscriptApi()
        self._language_priority = language_priority or DEFAULT_LANGUAGE_PRIORITY

    def fetch(self, video_id: str) -> TranscriptResult | None:
        from youtube_transcript_api import (
            CouldNotRetrieveTranscript,
            NoTranscriptFound,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
            VideoUnplayable,
        )

        self.last_failure_reason = None
        try:
            transcript_list = self._api.list(video_id)
        except TranscriptsDisabled:
            self.last_failure_reason = "disabled"
            return None
        except (VideoUnavailable, VideoUnplayable):
            self.last_failure_reason = "video_unavailable"
            return None
        except RequestBlocked:
            # Covers IpBlocked too (RequestBlocked subclass) -- this is the
            # "we're being rate-limited, stop hammering" signal.
            self.last_failure_reason = "blocked"
            return None
        except CouldNotRetrieveTranscript:
            self.last_failure_reason = "other"
            return None

        # Prefer a manually-created transcript over an auto-generated one,
        # in our language priority order; fall back to an auto-generated
        # one in those languages if no manual transcript exists.
        try:
            transcript = transcript_list.find_manually_created_transcript(self._language_priority)
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(self._language_priority)
            except NoTranscriptFound:
                self.last_failure_reason = "no_captions_in_preferred_languages"
                return None

        try:
            fetched = transcript.fetch()
        except RequestBlocked:
            # .fetch() hits a different endpoint than .list() above -- a
            # block can manifest here even when list() just succeeded.
            # Confirmed for real: a 170-video run got IP-blocked partway
            # through, and every failure after that point came from here.
            self.last_failure_reason = "blocked"
            return None
        except CouldNotRetrieveTranscript:
            self.last_failure_reason = "fetch_failed"
            return None

        segments = [
            TranscriptSegment(text=s.text, start=s.start, duration=s.duration) for s in fetched
        ]
        text = " ".join(s.text.strip() for s in fetched if s.text.strip())
        if not text:
            self.last_failure_reason = "empty_transcript"
            return None

        return TranscriptResult(
            text=text,
            language_code=transcript.language_code,
            is_generated=transcript.is_generated,
            provider_name="youtube-transcript-api",
            segments=segments,
        )
