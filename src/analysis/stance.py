"""Stance detection interface (docs/analysis_layer_spec.md §1.4, §2.1, §4 pass 2).

StanceDetector is the swap boundary -- same pattern as
entities.EntityExtractor / collection.youtube.transcript.TranscriptProvider:
today's implementation (AnthropicStanceDetector) calls the Anthropic API; a
future local model or different provider becomes just another implementation
of this interface, and nothing in stance_storage.py or detect_stance.py needs
to change.

Stance is scored per (item, entity), not per item -- the same text can be
positive toward one entity and negative toward another (§1.4), so `detect`
takes the specific entities the item references and returns one
StanceResult per entity, judged independently.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Per the spec's stance ontology (§1.4). "neutral" also covers incidental/
# factual mentions with no real evaluative stance -- the most common case.
POLARITIES = ("positive", "negative", "neutral")


@dataclass
class EntityRef:
    """One entity an item references -- enough context for the model to
    judge stance toward it without needing the full entity record."""

    entity_id: str
    canonical_name: str
    surface_form: str


@dataclass
class StanceResult:
    entity_id: str
    polarity: str
    strength: float
    confidence: float


_SYSTEM_PROMPT = """You are a stance detection system for a coordinated-inauthentic-behavior \
(CIB) detection pipeline. The content you analyze is social media text in Hindi, Urdu, English, \
and code-mixed combinations of these -- spanning Devanagari script, Arabic/Nastaliq script (Urdu), \
and Latin script (including Romanized/transliterated Hindi and Urdu, e.g. "modi sarkar" or \
"pakistan zindabad"). Treat all of these as first-class input, not noise.

You will be given one piece of text and a numbered list of entities that text references. For EACH \
entity, independently determine the STANCE THE AUTHOR of the text takes toward that specific entity \
-- not the stance of anyone the author is quoting, and not your own opinion of the entity.

An author can hold opposite stances toward different entities in the same text -- e.g. praising one \
country while condemning another, or defending one leader while attacking a rival. Judge each entity \
strictly on its own terms; do not let the stance toward one entity bleed into your read of another.

For each entity, report:
- "polarity": one of "positive", "negative", "neutral". "neutral" covers incidental or purely \
factual mentions with no real evaluative stance -- this is the MOST COMMON case. Do not force a \
polarity onto a mention that is merely descriptive or where the entity is background context.
- "strength": float 0.0-1.0, how intensely that polarity is expressed (0 = barely perceptible lean, \
1 = maximally forceful praise or condemnation). A neutral polarity should carry a low strength, near 0.
- "confidence": float 0.0-1.0, your confidence in this specific polarity/strength call. Lower this \
for sarcasm, irony, ambiguous code-mixed phrasing, or any case where the surface sentiment could \
reasonably be read either way. This is a best-effort signal, not ground truth -- an honest low \
confidence is strongly preferred over a falsely certain guess.

Sarcasm and irony are common in this corpus (e.g. text that praises an entity in a way clearly \
meant to mock it, or uses exaggerated enthusiasm to attack). Read for authorial intent, not literal \
word polarity, but lower confidence whenever the read is genuinely ambiguous rather than guessing.

Respond with ONLY a JSON array, no other text, markdown formatting, or explanation. Produce exactly \
one element per entity in the input list, in the SAME ORDER, each carrying that entity's number:
{"index": <integer, the entity's number from the input list>, "polarity": "positive"|"negative"|\
"neutral", "strength": <float 0.0-1.0>, "confidence": <float 0.0-1.0>}"""


class StanceDetector(ABC):
    """Swap boundary for stance detection. Implementations should raise on
    genuine failures (bad API key, network down) but return an empty list
    only when `entities` is empty or `text` is blank -- otherwise a
    best-effort result (even low-confidence) is expected for every entity
    requested.
    """

    @abstractmethod
    def detect(
        self, text: str, entities: list[EntityRef], context: dict | None = None
    ) -> list[StanceResult]:
        """Detect the author's stance toward each of `entities` in `text`.
        `context` may carry hints (language_detected, script, source_type)
        that help disambiguate short or ambiguous text; implementations may
        ignore keys they don't use. Returns at most one StanceResult per
        entity in `entities`; callers must handle a shorter result list
        (the model omitted an entity) as a detection gap, not an error.
        """
        raise NotImplementedError


def _build_user_prompt(text: str, entities: list[EntityRef], context: dict | None) -> str:
    hints = []
    if context:
        if context.get("language_detected"):
            hints.append(f"detected language: {context['language_detected']}")
        if context.get("script"):
            hints.append(f"detected script: {context['script']}")
        if context.get("source_type"):
            hints.append(f"source type: {context['source_type']}")
    hint_line = f"Context hints ({', '.join(hints)}):\n" if hints else ""

    entity_lines = "\n".join(
        f'{i}. {e.canonical_name} (as mentioned in this text: "{e.surface_form}")'
        for i, e in enumerate(entities, start=1)
    )
    return f'{hint_line}Entities to assess:\n{entity_lines}\n\nText:\n"""\n{text}\n"""'


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_response(raw: str, entities: list[EntityRef]) -> list[StanceResult]:
    """Parse the model's JSON array response, mapping each row's 1-based
    "index" back to the corresponding EntityRef's entity_id. Malformed rows
    or out-of-range indices are dropped rather than crashing the run --
    callers see the gap via a shorter-than-requested result list (see
    StanceDetector.detect docstring) and report it, they don't except on it.
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

    results = []
    for row in data:
        try:
            index = int(row["index"])
            if not (1 <= index <= len(entities)):
                continue
            polarity = str(row["polarity"]).lower()
            if polarity not in POLARITIES:
                continue
            results.append(
                StanceResult(
                    entity_id=entities[index - 1].entity_id,
                    polarity=polarity,
                    strength=_clamp01(float(row["strength"])),
                    confidence=_clamp01(float(row["confidence"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # one malformed element shouldn't drop the rest
    return results


class AnthropicStanceDetector(StanceDetector):
    """Detects per-entity stance via the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def detect(
        self, text: str, entities: list[EntityRef], context: dict | None = None
    ) -> list[StanceResult]:
        if not text or not text.strip() or not entities:
            return []
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(text, entities, context)}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        return _parse_response(raw, entities)
