"""POST /api/ask -- natural-language question -> structured query -> answer.

Local model only (Ollama, see serve.config.OLLAMA_MODEL), one call per
question, free. The model's ONLY job is to pick one of a small fixed set
of query_types and
extract plain-text params (an entity name, an author name, a topic phrase)
from the question -- it never sees the database, never generates SQL, and
never gets taught the id vocabulary. Every query_type maps 1:1 to an
existing, already-parameterized query function in queries.py/profiles.py;
name->id resolution (resolve.py) sits between the model's output and those
functions. This keeps the whole endpoint "safe pre-defined queries only" by
construction: an unrecognized or malformed model response degrades to
query_type "unsupported", never to an arbitrary query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import duckdb

from serve import queries, resolve
from serve.config import OLLAMA_HOST, OLLAMA_MODEL

_MAX_OUTPUT_TOKENS = 512

_VALID_QUERY_TYPES = {
    "consistent_actors",
    "author_stance_on_entity",
    "entity_timeline",
    "author_profile",
    "topic_coordination",
    "unsupported",
}
_VALID_DIRECTIONS = {"positive", "negative", "both"}
_VALID_BUCKETS = {"day", "week", "month"}

_SYSTEM_PROMPT = """You are a query router for a database analyzing coordinated activity across Telegram channels and YouTube videos discussing India-Pakistan geopolitics. You do NOT answer questions yourself and you NEVER write SQL. Your only job is to read a natural-language question and translate it into exactly ONE structured query from the fixed list below.

Available query_type values:

1. "consistent_actors" -- who consistently pushes FOR or AGAINST a given entity (e.g. "who is most anti-India?", "who consistently defends Pakistan?", "who are India's biggest supporters?").
   Params: entity_name (required -- the entity being discussed, e.g. "India"), direction ("positive" | "negative" | "both", default "both" -- "anti-X" / "against X" / "attacks X" / "criticizes X" means "negative"; "pro-X" / "defends X" / "supports X" means "positive").

2. "author_stance_on_entity" -- one specific channel/author's stance toward one specific entity (e.g. "what is Resonant News's stance on Pakistan?", "how does @News_Pakistan talk about India?").
   Params: author_name (required), entity_name (required).

3. "entity_timeline" -- how stance toward an entity has changed over time (e.g. "how has sentiment toward India changed?").
   Params: entity_name (required), bucket ("day" | "week" | "month", default "week").

4. "author_profile" -- one author's full stance vector across every entity they discuss (e.g. "what does Resonant News talk about?").
   Params: author_name (required).

5. "topic_coordination" -- which channels are coordinating (reposting the same content) around a topic or event (e.g. "show me coordination around Operation Sindoor", "is there coordinated activity about the Indus Waters Treaty?").
   Params: topic_query (required -- the topic/event phrase, in the questioner's own words).

6. "unsupported" -- the question does not fit any of the above (asks for something outside this database, or is too vague to extract required params from).
   Params: reason (required -- one short sentence explaining why).

Respond with ONLY a JSON object, no prose, no markdown code fences, matching exactly this shape (omit any key that does not apply to the chosen query_type -- do not include empty strings for irrelevant fields):
{"query_type": "...", "entity_name": "...", "author_name": "...", "topic_query": "...", "bucket": "...", "direction": "...", "reason": "..."}

Use entity/author names exactly as a person would say them (e.g. "India", "Pakistan", "Resonant News") -- you have no access to the database and must NOT invent ids.
"""


@dataclass
class AskResult:
    question: str
    intent: dict = field(default_factory=dict)
    result: dict | list | None = None
    summary: str = ""
    error: str | None = None


def _call_router(question: str) -> dict:
    import ollama

    # Explicit host= (rather than the module-level ollama.chat() convenience
    # function, which reads OLLAMA_HOST only once at import time) so this
    # always reflects the current env var -- OLLAMA_HOST unset -> local
    # `ollama serve` default; OLLAMA_HOST=http://ollama:11434 -> the
    # docker-compose `ollama` service, set in docker-compose.yml.
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        format="json",
        options={"num_predict": _MAX_OUTPUT_TOKENS},
    )
    raw = response["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"query_type": "unsupported", "reason": "the local model returned malformed output for this question"}
    if not isinstance(data, dict) or data.get("query_type") not in _VALID_QUERY_TYPES:
        return {"query_type": "unsupported", "reason": "the model's response did not match a supported query type"}
    return data


def _pick(intent: dict, key: str, valid: set[str], default: str) -> str:
    value = intent.get(key)
    return value if isinstance(value, str) and value in valid else default


def _unsupported(question: str, intent: dict, message: str) -> AskResult:
    return AskResult(question=question, intent=intent, result=None, summary=f"I can't answer that -- {message}", error=message)


def _summarize_consistent_actors(canonical_name: str, direction: str, result: dict) -> str:
    positive, negative = result["consistently_positive"], result["consistently_negative"]
    parts = []
    if direction in ("negative", "both"):
        if negative:
            top = negative[0]
            parts.append(
                f"most consistently NEGATIVE toward {canonical_name}: {top['author_id']} "
                f"(net_stance={top['net_stance']:+.2f}, consistency={top['stance_consistency']:.2f}, volume={top['volume']})"
            )
        else:
            parts.append(f"no author clears the ranking bar for negative stance toward {canonical_name}")
    if direction in ("positive", "both"):
        if positive:
            top = positive[0]
            parts.append(
                f"most consistently POSITIVE toward {canonical_name}: {top['author_id']} "
                f"(net_stance={top['net_stance']:+.2f}, consistency={top['stance_consistency']:.2f}, volume={top['volume']})"
            )
        else:
            parts.append(f"no author clears the ranking bar for positive stance toward {canonical_name}")
    return "; ".join(parts)


def _summarize_author_stance(author_match: dict, entity_match: dict, entry: dict | None) -> str:
    if entry is None:
        return f"{author_match['display_name']} ({author_match['author_id']}) has no recorded stance toward {entity_match['canonical_name']}."
    net_stance = entry["net_stance"]
    lean = "positive" if net_stance > 0.15 else "negative" if net_stance < -0.15 else "neutral/mixed"
    return (
        f"{author_match['display_name']} ({author_match['author_id']}) leans {lean} toward {entity_match['canonical_name']} "
        f"(net_stance={net_stance:+.2f}, consistency={entry['stance_consistency']:.2f}, volume={entry['volume']} items: "
        f"{entry['positive_count']} positive / {entry['negative_count']} negative / {entry['neutral_count']} neutral)."
    )


def _summarize_timeline(canonical_name: str, bucket: str, result: dict) -> str:
    timeline = result["timeline"]
    if not timeline:
        return f"No stance timeline data for {canonical_name}."
    total = sum(b["positive"] + b["negative"] + b["neutral"] for b in timeline)
    return f"{canonical_name}: {total} stance-bearing items across {len(timeline)} {bucket} buckets, from {timeline[0]['bucket_start']} to {timeline[-1]['bucket_start']}."


def _summarize_author_profile(match: dict, result: dict) -> str:
    if not result["stance_vector"]:
        return f"{match['display_name']} ({match['author_id']}) has no stance data."
    top = result["stance_vector"][:3]
    top_str = ", ".join(f"{e['canonical_name']} ({e['net_stance']:+.2f})" for e in top)
    return f"{match['display_name']} ({match['author_id']}), {result['item_count']} items. Top entities by |stance|*volume: {top_str}."


def _summarize_topic_coordination(topic_query: str, result: dict) -> str:
    clusters = result["clusters"]
    if not clusters:
        return f"No coordinated clusters found matching {topic_query!r}."
    top = clusters[0]
    return (
        f"Found {len(clusters)} coordinated cluster(s) matching {topic_query!r}; the largest spans "
        f"{top['size']} items across {top['distinct_authors']} distinct authors (sample: {top['sample_text']!r})."
    )


def answer(con: duckdb.DuckDBPyConnection, question: str) -> AskResult:
    question = question.strip()
    if not question:
        return _unsupported(question, {}, "the question was empty")

    intent = _call_router(question)
    query_type = intent.get("query_type")

    if query_type == "unsupported":
        return _unsupported(question, intent, intent.get("reason") or "this question doesn't map to a supported query")

    if query_type == "consistent_actors":
        entity_name = intent.get("entity_name")
        if not entity_name:
            return _unsupported(question, intent, "no entity name was extracted from the question")
        match = resolve.resolve_entity(con, entity_name)
        if match is None:
            return _unsupported(question, intent, f"no entity found matching {entity_name!r}")
        direction = _pick(intent, "direction", _VALID_DIRECTIONS, "both")
        result = queries.get_entity_authors(con, match["entity_id"], limit=5, min_volume=5)
        assert result is not None
        return AskResult(question, intent, result, _summarize_consistent_actors(match["canonical_name"], direction, result))

    if query_type == "author_stance_on_entity":
        author_name, entity_name = intent.get("author_name"), intent.get("entity_name")
        if not author_name or not entity_name:
            return _unsupported(question, intent, "an author and an entity were both required but not both extracted")
        author_match = resolve.resolve_author(con, author_name)
        if author_match is None:
            return _unsupported(question, intent, f"no author found matching {author_name!r}")
        entity_match = resolve.resolve_entity(con, entity_name)
        if entity_match is None:
            return _unsupported(question, intent, f"no entity found matching {entity_name!r}")
        profile = queries.get_author_profile(con, author_match["author_id"], limit=5000)
        assert profile is not None
        entry = next((e for e in profile["stance_vector"] if e["entity_id"] == entity_match["entity_id"]), None)
        result = {"author": author_match, "entity": entity_match, "stance": entry}
        return AskResult(question, intent, result, _summarize_author_stance(author_match, entity_match, entry))

    if query_type == "entity_timeline":
        entity_name = intent.get("entity_name")
        if not entity_name:
            return _unsupported(question, intent, "no entity name was extracted from the question")
        match = resolve.resolve_entity(con, entity_name)
        if match is None:
            return _unsupported(question, intent, f"no entity found matching {entity_name!r}")
        bucket = _pick(intent, "bucket", _VALID_BUCKETS, "week")
        result = queries.get_entity_timeline(con, match["entity_id"], bucket)
        assert result is not None
        return AskResult(question, intent, result, _summarize_timeline(match["canonical_name"], bucket, result))

    if query_type == "author_profile":
        author_name = intent.get("author_name")
        if not author_name:
            return _unsupported(question, intent, "no author name was extracted from the question")
        match = resolve.resolve_author(con, author_name)
        if match is None:
            return _unsupported(question, intent, f"no author found matching {author_name!r}")
        result = queries.get_author_profile(con, match["author_id"], limit=10)
        assert result is not None
        return AskResult(question, intent, result, _summarize_author_profile(match, result))

    if query_type == "topic_coordination":
        topic_query = intent.get("topic_query")
        if not topic_query:
            return _unsupported(question, intent, "no topic/keyword was extracted from the question")
        result = queries.get_topic_coordination(con, topic_query, limit=5)
        return AskResult(question, intent, result, _summarize_topic_coordination(topic_query, result))

    return _unsupported(question, intent, "unrecognized query type")
