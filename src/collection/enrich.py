"""Shared content-enrichment step: text normalization, language + script
detection, and content hashing.

Applied automatically to every Item at construction time (see
Item.model_post_init in schema.py) -- no collector calls this directly.
This is what makes language_detected, script, text_normalized, and
content_hashes populate for Telegram today, and for every future source
collector (YouTube, news) without repeating this logic per-source.
"""

from __future__ import annotations

import hashlib
import re

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic results across runs
except ImportError:  # pragma: no cover -- collect group not installed
    detect = None
    LangDetectException = Exception  # type: ignore[assignment,misc]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

# langdetect is unreliable below this length -- garbage in, garbage out.
_MIN_LANGDETECT_CHARS = 10

# Unicode script block ranges relevant to Hindi/English/Urdu CIB detection
# (doc §1 `script`: latin | devanagari | arabic | mixed). The Arabic block
# covers Urdu -- Urdu-specific letters (e.g. ٹ ڈ ڑ ں ے) all fall within
# U+0600-06FF, so no separate Urdu range is needed.
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
_LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]

# A script "wins" outright above this share of classifiable letters;
# otherwise the text counts as mixed-script.
_DOMINANT_THRESHOLD = 0.85


def normalize_text(text: str | None) -> str | None:
    """Lowercase, collapse whitespace, strip URLs. None if nothing meaningful remains."""
    if not text:
        return None
    stripped = _URL_RE.sub(" ", text)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip().lower()
    return collapsed or None


def detect_language(text: str | None) -> str | None:
    """ISO 639-1 language code via langdetect. None for missing/too-short/undetectable text."""
    if not text or len(text) < _MIN_LANGDETECT_CHARS or detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _char_script(ch: str) -> str | None:
    code = ord(ch)
    if _DEVANAGARI_RANGE[0] <= code <= _DEVANAGARI_RANGE[1]:
        return "devanagari"
    if any(lo <= code <= hi for lo, hi in _ARABIC_RANGES):
        return "arabic"
    if any(lo <= code <= hi for lo, hi in _LATIN_RANGES):
        return "latin"
    return None


def detect_script(text: str | None) -> str | None:
    """Dominant Unicode script block: latin / devanagari / arabic / mixed. None if no classifiable letters."""
    if not text:
        return None
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        script = _char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return None
    dominant_script, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    if dominant_count / total >= _DOMINANT_THRESHOLD:
        return dominant_script
    return "mixed"


def compute_text_hash(text_normalized: str | None) -> str | None:
    """sha256 of the normalized text -- feeds the near_duplicate_text derived edge (doc §6)."""
    if not text_normalized:
        return None
    return hashlib.sha256(text_normalized.encode("utf-8")).hexdigest()


def compute_media_hash(media: list[dict] | None) -> str | None:
    """Hash of the source's own file id(s) for this item's media -- NOT a
    content hash. No API we collect from exposes a true content hash
    without downloading the file (doc §3, hashes+refs only / fetch on
    demand); this hashes the stable native identifiers we already have
    (`source_ref`, e.g. Telegram's document/photo id) so exact re-posts of
    the same source file are still linkable pre-download. Feeds the
    shared_media derived edge (doc §6).
    """
    if not media:
        return None
    refs = [str(m["source_ref"]) for m in media if m.get("source_ref")]
    if not refs:
        return None
    return hashlib.sha256(":".join(refs).encode("utf-8")).hexdigest()


def enrich_item_fields(text: str | None, media: list[dict] | None) -> dict:
    """Compute every enrichment field for one item's text + media in one place."""
    text_normalized = normalize_text(text)
    return {
        "text_normalized": text_normalized,
        "language_detected": detect_language(text),
        "script": detect_script(text),
        "content_hashes": {
            "text_hash": compute_text_hash(text_normalized),
            "media_hash": compute_media_hash(media),
        },
    }
