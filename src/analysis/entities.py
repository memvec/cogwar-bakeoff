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

from analysis.gemini_retry import call_with_retry

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
        # (input_tokens, output_tokens) from the most recent extract() call --
        # diagnostic only, not part of the EntityExtractor contract, but kept
        # in the SAME shape as GeminiEntityExtractor's so callers (cost
        # tracking in extract_entities.py) can read it uniformly regardless
        # of provider.
        self.last_usage_tokens: tuple[int, int] | None = None

    def extract(self, text: str, context: dict | None = None) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(text, context)}],
        )
        self.last_usage_tokens = (response.usage.input_tokens, response.usage.output_tokens)
        raw = "".join(block.text for block in response.content if block.type == "text")
        return _parse_response(raw)


class GeminiEntityExtractor(EntityExtractor):
    """Extracts + resolves entity mentions via the Gemini API.

    Same prompt (_SYSTEM_PROMPT / _build_user_prompt) and same response
    parsing (_parse_response) as AnthropicEntityExtractor -- only the API
    call differs, which is the whole point of the swap boundary: a
    head-to-head comparison (analysis/compare_providers.py) is comparing
    the MODELS, not different prompting strategies.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048) -> None:
        from google import genai
        from google.genai import types

        # Explicit per-request timeout: a call that never gets a response at
        # all (a genuine network hang, distinct from a 429/503 the API
        # actually returns) is NOT caught by gemini_retry.py's retry logic,
        # since there's no exception to retry on -- observed in practice as
        # a multi-batch stall with connections sitting open and 0% CPU.
        # Bounding it here means a hang surfaces as a normal timeout
        # exception, which the per-item failure handling in
        # extract_entities.py/detect_stance.py already knows how to skip.
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000))
        self._model = model
        # Gemini 3.x Flash spends some of max_output_tokens on internal
        # "thoughts" before any visible text (observed ~75 tokens even for a
        # trivial prompt) -- a higher default than Claude's 1024 leaves
        # headroom for that overhead plus the actual JSON.
        self._max_tokens = max_tokens
        # (input_tokens, output_tokens) from the most recent extract() call,
        # output including "thoughts" tokens (billed as output). Diagnostic
        # only -- not part of the EntityExtractor contract -- set after
        # every call so cost tracking doesn't need a second, wasted API call.
        self.last_usage_tokens: tuple[int, int] | None = None

    def extract(self, text: str, context: dict | None = None) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        from google.genai import types

        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=_build_user_prompt(text, context),
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=self._max_tokens,
                ),
            )
        )
        usage = response.usage_metadata
        self.last_usage_tokens = (
            (usage.prompt_token_count or 0) if usage else 0,
            ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )
        return _parse_response(response.text or "")


class OllamaEntityExtractor(EntityExtractor):
    """Extracts + resolves entity mentions via a local Ollama model.

    Same prompt (_SYSTEM_PROMPT / _build_user_prompt) and same response
    parsing (_parse_response) as AnthropicEntityExtractor/GeminiEntityExtractor
    -- only the call differs, same reasoning as GeminiEntityExtractor's
    docstring: a head-to-head comparison is comparing MODELS, not prompting
    strategies. No API key, no network egress, no cost.
    """

    #: Hard cap on generated tokens, same role as AnthropicEntityExtractor's
    #: max_tokens/GeminiEntityExtractor's max_output_tokens -- confirmed for
    #: real that omitting this is dangerous: a local model can drop into a
    #: degenerate repetition loop and grind on with no natural stop
    #: condition, observed for real as a single call running 17+ minutes
    #: (vs. a normal few seconds to ~1 minute) before being killed. The
    #: expected output here is a short JSON array, nowhere near this cap in
    #: the normal case.
    _MAX_OUTPUT_TOKENS = 1024

    def __init__(self, model: str, api_key: str | None = None) -> None:
        # api_key accepted (and ignored) only so this fits build_extractor's
        # shared (provider, api_key, model) factory signature alongside the
        # cloud providers -- Ollama is local, nothing to authenticate.
        self._model = model
        # Always None: there's no token-usage/cost concept for a local
        # model, and callers (extract_entities.py's cost tracking) already
        # guard on `is not None` before touching a CostTracker.
        self.last_usage_tokens: tuple[int, int] | None = None

    def extract(self, text: str, context: dict | None = None) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        import ollama

        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(text, context)},
            ],
            options={"num_predict": self._MAX_OUTPUT_TOKENS},
        )
        return _parse_response(response["message"]["content"])


def build_extractor(provider: str, api_key: str, model: str) -> EntityExtractor:
    """Provider-selection factory: construct the EntityExtractor implementation
    named by `provider` ("anthropic", "gemini", or "ollama"). The swap
    boundary is the ABC above; this is just a convenience so callers don't
    hardcode a class name -- callers own reading api_key/model out of config
    for whichever provider they picked (api_key is ignored for "ollama").
    """
    if provider == "anthropic":
        return AnthropicEntityExtractor(api_key=api_key, model=model)
    if provider == "gemini":
        return GeminiEntityExtractor(api_key=api_key, model=model)
    if provider == "ollama":
        return OllamaEntityExtractor(model=model)
    raise ValueError(f"Unknown provider: {provider!r} (expected 'anthropic', 'gemini', or 'ollama')")


def build_extractor_pool(provider: str, api_key: str, model: str, size: int) -> list[EntityExtractor]:
    """`size` independent instances, for safe concurrent use across worker
    threads. Each instance's last_usage_tokens is per-instance mutable state
    (see GeminiEntityExtractor/AnthropicEntityExtractor) -- two threads
    calling .extract() on the SAME instance at once would race on it, so
    callers must give each concurrent worker its own instance from this
    pool rather than sharing one."""
    return [build_extractor(provider, api_key=api_key, model=model) for _ in range(size)]
