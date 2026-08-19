"""Entity extraction interface (docs/analysis_layer_spec.md §1.1, §1.2, §4 pass 1).

EntityExtractor is the swap boundary -- same pattern as
collection.youtube.transcript.TranscriptProvider: today's implementation
(AnthropicEntityExtractor) calls the Anthropic API; a future local model or
different provider becomes just another implementation of this interface,
and nothing in resolution.py or extract_entities.py needs to change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Entity types per the spec's entity ontology (§1.1).
ENTITY_TYPES = ("country", "person", "brand", "org", "topic", "other")

_SYSTEM_PROMPT = """You are an entity extraction system for a coordinated-inauthentic-behavior \
(CIB) detection pipeline. The content you analyze is social media text in Hindi, Urdu, English, \
and code-mixed combinations of these -- spanning Devanagari script, Arabic/Nastaliq script (Urdu), \
and Latin script (including Romanized/transliterated Hindi and Urdu, e.g. "modi sarkar" or \
"pakistan zindabad"). Treat all of these as first-class input, not noise.

Given one piece of text, extract entities that are the TARGET of discussion or narrative framing \
-- countries, persons, organizations, brands, and named topics/events that the text is substantively \
ABOUT. Do not extract every noun or incidental mention; only extract entities the text is actually \
discussing, praising, criticizing, or framing.

For each entity, resolve ALL surface-form variants to ONE canonical name. The following must all \
map to the SAME canonical_name when they refer to the same real-world entity:
- different spellings and transliterations (e.g. "Mumbai" / "Bombay")
- honorifics and titles attached or removed (e.g. "PM Modi" / "Modi" / "Narendra Modi" / "Sheikh Hasina")
- script variants (Devanagari / Arabic / Latin / Romanized) of the same name
- abbreviations and acronyms (e.g. "PoK" / "Pakistan-occupied Kashmir")
- emoji and flag symbols that denote an entity (e.g. \U0001F1EE\U0001F1F3 = India, \U0001F1F5\U0001F1F0 = Pakistan)

canonical_name should always be written in English (Latin script), using the most standard/common \
English form of the name, regardless of what script or language the surface_form appeared in.

Respond with ONLY a JSON array, no other text, markdown formatting, or explanation. Each element:
{"surface_form": "<exact text as it appeared in the input>", "entity_type_guess": "<one of: \
country, person, brand, org, topic, other>", "canonical_name": "<resolved canonical name, in \
English>", "confidence": <float 0.0-1.0>}

If no qualifying entities are found, respond with exactly: []"""


@dataclass
class EntityMention:
    surface_form: str
    entity_type_guess: str
    canonical_name: str
    confidence: float


class EntityExtractor(ABC):
    """Swap boundary for entity extraction. Implementations should raise on
    genuine failures (bad API key, network down) but return an empty list
    for "no entities found" -- that's a normal outcome, not an error.
    """

    @abstractmethod
    def extract(self, text: str, context: dict | None = None) -> list[EntityMention]:
        """Extract entity mentions from `text`. `context` may carry hints
        (e.g. language_detected, script, source_type) that help a short or
        ambiguous text get resolved correctly; implementations may ignore
        keys they don't use.
        """
        raise NotImplementedError


def _build_user_prompt(text: str, context: dict | None) -> str:
    hints = []
    if context:
        if context.get("language_detected"):
            hints.append(f"detected language: {context['language_detected']}")
        if context.get("script"):
            hints.append(f"detected script: {context['script']}")
        if context.get("source_type"):
            hints.append(f"source type: {context['source_type']}")
    hint_line = f"Context hints ({', '.join(hints)}):\n" if hints else ""
    return f'{hint_line}Text:\n"""\n{text}\n"""'


def _parse_response(raw: str) -> list[EntityMention]:
    """Parse the model's JSON array response. Malformed output is treated as
    zero entities found rather than a crash -- extraction quality issues are
    surfaced via the "items with zero entities" report, not exceptions.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json")
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    mentions = []
    for row in data:
        try:
            mentions.append(
                EntityMention(
                    surface_form=str(row["surface_form"]),
                    entity_type_guess=str(row["entity_type_guess"]).lower(),
                    canonical_name=str(row["canonical_name"]),
                    confidence=float(row["confidence"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # one malformed element shouldn't drop the rest
    return mentions


class AnthropicEntityExtractor(EntityExtractor):
    """Extracts + resolves entity mentions via the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def extract(self, text: str, context: dict | None = None) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(text, context)}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        return _parse_response(raw)
