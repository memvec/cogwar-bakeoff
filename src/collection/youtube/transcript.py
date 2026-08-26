"""Transcript provider interface for YouTube videos.

TranscriptProvider is the swap boundary: today's implementation
(YoutubeTranscriptApiProvider) scrapes existing captions via the
youtube-transcript-api library. A future Whisper-based or commercial-API
provider becomes just another implementation of this same interface --
nothing in enrich_transcripts.py or youtube/mapping.py needs to change.
"""

from __future__ import annotations

import tempfile
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter


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


_REQUEST_TIMEOUT_SECONDS = 30.0


class _TimeoutHTTPAdapter(HTTPAdapter):
    """Forces a timeout on every request made through this adapter.

    youtube-transcript-api's default requests.Session has no timeout at
    all, so a connection that stalls without erroring (not a 4xx/5xx, just
    silence) blocks forever -- confirmed for real: a run hung for 13+
    minutes on one video with only ~8s of actual CPU time used, i.e.
    parked in I/O wait indefinitely. Same failure mode as the Gemini
    client's missing http timeout hit earlier in this project (see
    analysis/config.py history) -- fixed there with an explicit timeout,
    fixed here the same way since youtube-transcript-api's http_client is
    just a plain requests.Session we can swap in with a timeout-enforcing
    adapter mounted on it.
    """

    def __init__(self, *args, timeout: float = _REQUEST_TIMEOUT_SECONDS, **kwargs) -> None:
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


class YoutubeTranscriptApiProvider(TranscriptProvider):
    """Scrapes existing (manual or auto-generated) YouTube captions via
    youtube-transcript-api. No audio processing -- only works for videos
    that already have captions in one of `language_priority`.
    """

    def __init__(self, language_priority: list[str] | None = None) -> None:
        from youtube_transcript_api import YouTubeTranscriptApi

        session = requests.Session()
        adapter = _TimeoutHTTPAdapter()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        self._api = YouTubeTranscriptApi(http_client=session)
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
        except requests.exceptions.RequestException:
            # Timeout (from the adapter above) or any other connection-level
            # failure -- treat like any other "couldn't get this one,
            # move on" outcome rather than crashing the whole run; the
            # TranscriptProvider contract requires fetch() to never raise.
            self.last_failure_reason = "timeout_or_network_error"
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


DEFAULT_WHISPER_MODEL_SIZE = "small"


class WhisperTranscriptProvider(TranscriptProvider):
    """ASR-based transcription via yt-dlp (audio download) + faster-whisper
    (local, offline speech-to-text) -- for videos with no existing captions
    at all, where YoutubeTranscriptApiProvider has nothing to scrape.
    Unlike that provider, this one costs real local compute per video (audio
    download + model inference) rather than a cheap HTTP call, so callers
    doing a large run should budget wall-clock time accordingly (see
    last_timing_seconds below).

    Downloads audio to a temp file per video and deletes it immediately
    after transcription (success or failure) -- never accumulates audio on
    disk across a run, which matters at any real video count.
    """

    #: Best-effort diagnostic (not part of the required contract, see
    #: TranscriptProvider.last_failure_reason's docstring for the pattern):
    #: {"download_seconds": ..., "transcribe_seconds": ...} for the most
    #: recent fetch() call, so callers can report a download-vs-transcribe
    #: time breakdown without needing their own instrumentation.
    last_timing_seconds: dict[str, float] | None = None

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
        download_dir: Path | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        """cpu_threads caps CTranslate2's CPU thread pool for this model --
        left None uses faster-whisper's own default (typically all cores).
        A throttled run should pass roughly half of os.cpu_count() here so
        transcription doesn't compete for every core on the machine it's
        running on.
        """
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads or 0
        )
        self._model_size = model_size
        self._download_dir = download_dir or Path(tempfile.gettempdir()) / "cogwar_whisper_audio"
        self._download_dir.mkdir(parents=True, exist_ok=True)

    def _download_audio(self, video_id: str) -> Path | None:
        """Best-effort audio-only download via yt-dlp. None on any failure -- a
        private/deleted/region-blocked video is a routine, expected outcome here.
        """
        import yt_dlp

        out_template = str(self._download_dir / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                downloaded_path = ydl.prepare_filename(info)
        except Exception:  # noqa: BLE001 -- yt-dlp raises many distinct error types for "couldn't get this video"; all are routine, expected outcomes here, not something that should crash a run
            return None

        path = Path(downloaded_path)
        return path if path.exists() else None

    def fetch(self, video_id: str) -> TranscriptResult | None:
        self.last_failure_reason = None
        self.last_timing_seconds = None

        t0 = time.monotonic()
        audio_path = self._download_audio(video_id)
        t1 = time.monotonic()
        if audio_path is None:
            self.last_failure_reason = "download_failed"
            self.last_timing_seconds = {"download_seconds": t1 - t0, "transcribe_seconds": 0.0}
            return None

        try:
            segments_iter, info = self._model.transcribe(str(audio_path), vad_filter=True)
            segments = list(segments_iter)
        except Exception as e:  # noqa: BLE001 -- local model/decode failure must not crash a run, same discipline as every other provider here
            self.last_failure_reason = f"whisper_failed: {type(e).__name__}: {e}"
            audio_path.unlink(missing_ok=True)
            return None
        t2 = time.monotonic()
        audio_path.unlink(missing_ok=True)
        self.last_timing_seconds = {"download_seconds": t1 - t0, "transcribe_seconds": t2 - t1}

        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        if not text:
            self.last_failure_reason = "empty_transcript"
            return None

        return TranscriptResult(
            text=text,
            language_code=info.language,
            is_generated=True,  # ASR output, never a human-authored caption
            provider_name=f"faster-whisper-{self._model_size}",
            segments=[
                TranscriptSegment(text=s.text, start=s.start, duration=s.end - s.start) for s in segments
            ],
        )


# --- Hallucination guard -----------------------------------------------
#
# faster-whisper's known degeneration failure mode on noisy/silent/long
# audio: instead of giving up, it can get "stuck" and emit the same token,
# short phrase, or whole segment over and over instead of real speech (seen
# for real in this project's own sample run -- a transcript that trailed
# off into the same Kannada character repeated dozens of times). Detecting
# this after the fact -- rather than trusting every ASR output -- matters
# because a hallucinated transcript would otherwise get fed to downstream
# entity/stance analysis as if it were real spoken content.

_MIN_LENGTH_FOR_DIVERSITY_CHECK = 100  # chars; shorter transcripts naturally have low diversity, not a signal
_DIVERSITY_WINDOW_CHARS = 150  # sliding-window size for the localized-degeneration check below
_MIN_UNIQUE_CHAR_RATIO = 0.12  # distinct chars / total chars in ANY window; below this = suspiciously repetitive
_MAX_CONSECUTIVE_NGRAM_REPEATS = 6  # same n-token phrase repeated back-to-back this many times = a loop
_NGRAM_SIZES = (1, 2, 3, 4)
_MAX_SEGMENT_REPEAT_COUNT = 5  # the exact same .transcribe() segment recurring this often anywhere = a stuck loop


def _longest_consecutive_ngram_run(tokens: list[str], n: int) -> int:
    """Longest run of the same n-token phrase repeating back-to-back
    (non-overlapping), checked from every starting offset -- e.g.
    tokens=[a,b,a,b,a,b], n=2 -> 3 (the phrase (a,b) repeats 3 times in a
    row). A naive adjacent-overlapping-window comparison (stride 1) can
    never detect this for n>=2, since shifting by 1 token misaligns the
    phrase boundary and no two adjacent windows are ever equal even when
    the phrase is repeating -- this checks non-overlapping windows spaced
    exactly `n` apart instead, which is what an actual repeat looks like.
    """
    if len(tokens) < n * 2:
        return 1
    best = 1
    for i in range(len(tokens) - n):
        current = tuple(tokens[i : i + n])
        run = 1
        j = i + n
        while j + n <= len(tokens) and tuple(tokens[j : j + n]) == current:
            run += 1
            j += n
        best = max(best, run)
    return best


def _min_window_unique_char_ratio(text: str, window: int) -> float:
    """Lowest unique-char-ratio among all `window`-sized slices of text.

    Catches LOCALIZED degeneration (e.g. a transcript that starts as real
    speech and only degrades into a repeated-character loop near the end)
    that a single whole-transcript ratio would dilute away -- confirmed for
    real on this project's own sample data: a transcript with real content
    up front and a 130-character single-character loop at the end had a
    whole-text ratio just above a naive threshold, but any window taken
    from inside the loop itself has a ratio near 1/window.
    """
    if len(text) < window:
        return len(set(text)) / len(text) if text else 1.0
    return min(len(set(text[i : i + window])) / window for i in range(0, len(text) - window + 1, window // 3))


def detect_whisper_hallucination(result: TranscriptResult) -> tuple[bool, str | None]:
    """Best-effort ASR-degeneration detector. Three independent checks, any
    one trips the guard -- (is_bad, reason); reason is None when not bad.

      1. A short n-gram (1-3 tokens) repeated many times *consecutively* --
         the clearest signature of a loop.
      2. Abnormally low character diversity across a long-enough transcript
         -- catches loops the n-gram check misses (e.g. a single character
         repeated with no word-breaking whitespace, so it's one giant
         "token" rather than many repeated ones).
      3. The exact same segment (one .transcribe() emission) recurring many
         times anywhere in the transcript, not just consecutively -- the
         model can drop into a loop, briefly recover, then relapse into
         the same line.

    False positives are possible on genuinely repetitive real speech (a
    chant, a slogan repeated for emphasis), but that's the safer failure
    mode here: better to under-trust a good transcript (the caller falls
    back to title/description) than to store hallucinated garbage as if it
    were real content feeding downstream analysis.
    """
    text = result.text or ""
    tokens = text.split()

    for n in _NGRAM_SIZES:
        run = _longest_consecutive_ngram_run(tokens, n)
        if run >= _MAX_CONSECUTIVE_NGRAM_REPEATS:
            return True, f"repeated {n}-gram loop ({run} consecutive repeats)"

    if len(text) >= _MIN_LENGTH_FOR_DIVERSITY_CHECK:
        worst_ratio = _min_window_unique_char_ratio(text, _DIVERSITY_WINDOW_CHARS)
        if worst_ratio < _MIN_UNIQUE_CHAR_RATIO:
            return True, f"abnormally low character diversity in a {_DIVERSITY_WINDOW_CHARS}-char window ({worst_ratio:.3f})"

    if result.segments:
        segment_texts = [s.text.strip() for s in result.segments if s.text.strip()]
        if segment_texts:
            top_text, top_count = Counter(segment_texts).most_common(1)[0]
            if top_count >= _MAX_SEGMENT_REPEAT_COUNT:
                return True, f"segment repeated {top_count} times: {top_text[:50]!r}"

    return False, None
