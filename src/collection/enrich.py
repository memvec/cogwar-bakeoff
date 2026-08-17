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

_URL_STRIP_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

# For extract_plaintext_entities: plain URLs, #hashtags, and @handles found
# in freeform text with no structured entity API (YouTube descriptions
# today; future news/factcheck article bodies). The handle regex requires a
# non-word/non-dot character (or start-of-string) immediately before the
# `@` -- without that, `@[\w.]+` also matches the domain half of an email
# address (e.g. "info.rjraunac@gmail.com" -> "@gmail.com"), a real false
# positive found in collected YouTube data.
_URL_FIND_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_HANDLE_RE = re.compile(r"(?<![\w.])@[\w.]+")

# langdetect is unreliable below this length -- garbage in, garbage out.
_MIN_LANGDETECT_CHARS = 10

# Unicode script block ranges relevant to Hindi/English/Urdu CIB detection
# (doc §1 `script`: latin | devanagari | arabic | mixed). The Arabic block
# covers Urdu -- Urdu-specific letters (e.g. ٹ ڈ ڑ ں ے) all fall within
# U+0600-06FF, so no separate Urdu range is needed.
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
_LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]

# A script "wins" outright only above this share of classifiable characters
# AND if every other script present has fewer than _MIXED_FLOOR_CHARS
# characters. The floor exists because a pure ratio, computed over an
# entire item's text, is easily diluted by volume: a YouTube video's
# title+description text is dominated by a long, largely-boilerplate
# English description (links, hashtags, "follow us" CTAs), which can drown
# out a short but genuinely code-mixed title (e.g. Devanagari+English in
# the same sentence) well past the ratio threshold even though ~40
# characters of a second script are clearly not incidental. The floor
# catches that: any script with a real, non-trivial character count forces
# "mixed" regardless of how much boilerplate volume surrounds it. Verified
# against 202 real collected items: only the 4 cases with a genuine
# secondary-script presence (>=20 chars) changed, all correctly latin/
# devanagari -> mixed; zero unwanted flips elsewhere.
_DOMINANT_THRESHOLD = 0.85
_MIXED_FLOOR_CHARS = 20


def normalize_text(text: str | None) -> str | None:
    """Lowercase, collapse whitespace, strip URLs. None if nothing meaningful remains."""
    if not text:
        return None
    stripped = _URL_STRIP_RE.sub(" ", text)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip().lower()
    return collapsed or None


def extract_plaintext_entities(text: str | None) -> dict:
    """@handles, #hashtags, and urls[] found in freeform text with no
    structured entity API (doc §1 `entities`). Telegram doesn't need this --
    Telethon's message.entities already gives offset-based entities -- but
    any source that hands us plain text (YouTube descriptions today, future
    news/factcheck article bodies) does.
    """
    if not text:
        return {"urls": [], "hashtags": [], "mentions": [], "handles": [], "named_entities": []}
    mentions = _HANDLE_RE.findall(text)
    return {
        "urls": _URL_FIND_RE.findall(text),
        "hashtags": _HASHTAG_RE.findall(text),
        "mentions": mentions,
        "handles": [m.lstrip("@") for m in mentions],
        "named_entities": [],  # NER not implemented at collection time (doc §0 principle 5 -- not collection's job)
    }


def detect_language(text: str | None, *, script: str | None = None) -> str | None:
    """ISO 639-1 language code via langdetect. None for missing/too-short/undetectable text.

    IMPORTANT -- best-effort only, not authoritative: langdetect is a
    single-language classifier and has no way to represent code-mixed text
    (e.g. Hindi+English in the same sentence, extremely common in this
    project's target content). Two known-unreliable cases:

    1. Romanized Hindi/Urdu (Latin script, South Asian language) is
       routinely misclassified as an unrelated language (verified: a
       Romanized Hindi/Urdu test string came back "sw" -- Swahili). There is
       no reliable signal to detect this failure mode from the output alone,
       so it is NOT specially handled here; script stays "latin" (correct),
       language_detected may simply be wrong. Treat it as a hint, not a fact.
    2. When `script` is "mixed", a single ISO code cannot honestly describe
       the text at all -- we don't report one (see below) rather than
       returning a falsely-confident guess.

    `script` should be the result of detect_script() on the same text, if
    already computed, so case 2 can be applied without re-detecting script.
    """
    if not text or len(text) < _MIN_LANGDETECT_CHARS or detect is None:
        return None
    if script == "mixed":
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
    """Dominant Unicode script block: latin / devanagari / arabic / mixed. None if no classifiable characters.

    Classifies every character whose codepoint falls in a known script
    range (see _char_script) -- NOT gated on str.isalpha(). Devanagari
    (and other Indic) combining vowel signs/virama are Unicode category
    Mn/Mc, so isalpha() is False for them even though they're clearly part
    of the script; gating on it silently undercounted Devanagari presence
    (verified against real collected data: e.g. "नक्सल" only counted its
    4 base consonants, not the vowel signs riding on them).
    """
    if not text:
        return None
    counts: dict[str, int] = {}
    for ch in text:
        script = _char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return None
    dominant_script, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    other_total = total - dominant_count
    if dominant_count / total >= _DOMINANT_THRESHOLD and other_total < _MIXED_FLOOR_CHARS:
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
    script = detect_script(text)
    return {
        "text_normalized": text_normalized,
        "language_detected": detect_language(text, script=script),
        "script": script,
        "content_hashes": {
            "text_hash": compute_text_hash(text_normalized),
            "media_hash": compute_media_hash(media),
        },
    }
