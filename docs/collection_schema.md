# Collection Layer — Canonical Schema Specification

**Status:** LOCKED v0.6 — §8, §9 resolved; §1 `source_type` extended with `channel`/`author`; §9 clarified — Telegram `account_created_at` is always `null` (platform limitation), estimate lives in `source_specific` only; §9 `observation` regained `view_count` for sources that expose it (e.g. YouTube) (see §8, §9)
**Scope:** Layer 1 (Collection) of the four-layer stack.
**Sources covered:** Telegram (public channels), YouTube (Data API v3), Fact-check + News (unstructured web).
**Design model:** Option A — single universal `items` model + separate `edges` model. Every collected object is an `item`; every relationship is an `edge`. The coordination graph = items (nodes) + edges.

---

## 0. Design principles (the rules the schema enforces)

1. **One canonical model, all sources normalize into it.** Rich API sources and sparse web sources map into the *same* `item` shape. Demonstrating this normalization is an explicit MVP capability.
2. **Graph-native from collection.** Edges are first-class and captured *at collection time*, not derived later. If a forward/reply/embed relationship is visible when we collect, we record the edge then.
3. **Store raw, always.** Every item keeps the full raw payload (JSON for APIs, HTML for web). Parsed fields are a convenience layer over the raw; we never discard the raw. Reason: we will want a field later we didn't parse now, and the source content may be gone.
4. **Provenance is not optional.** Every item records how/when/by-what it was collected, separately from when it was published. A government-facing system must be able to defend where every datum came from.
5. **Collection does not judge.** No coordination verdicts, no attribution, no labels-as-truth at this layer. Collection captures signal; detection/attribution/governance interpret it downstream. (Fact-check verdicts are captured as *source-asserted claims*, not as system truth — see §4.)
6. **Confidence travels with sparse data.** Web extraction is lossy; every extracted field from unstructured sources carries an extraction-confidence marker so downstream layers know how much to trust it.

---

## 1. Core `item` model (all sources)

Every collected object — a Telegram message, a YouTube video, a YouTube comment, a news article, a fact-check article — is one `item` row. Common core fields, populated by all sources where available:

| Field | Type | Notes |
|---|---|---|
| `item_id` | string (UUID) | Internal universal ID. Primary key. The graph node ID. |
| `source_type` | enum | `telegram` \| `youtube_video` \| `youtube_comment` \| `news` \| `factcheck` \| `channel` \| `author` — the last two are promoted author/channel nodes (§8.2, §9), generic across sources rather than source-prefixed. |
| `source_native_id` | string | The source's own ID (Telegram msg id, YT video/comment id, article canonical URL). |
| `parent_item_id` | string (UUID) \| null | For nested items (a YT comment's video; a reply's parent). Structural containment, not a graph edge. |
| `text` | string \| null | Primary textual content (message text, video title+description, article body, comment text). |
| `text_normalized` | string \| null | Cleaned/normalized text for hashing + dedup (lowercased, whitespace-collapsed, URLs canonicalized). Derived. |
| `language_detected` | string (ISO 639) \| null | Auto-detected. Distinct from source-declared language. |
| `language_declared` | string \| null | Source-declared language where available (YT `defaultLanguage`, article `lang`). |
| `script` | enum \| null | `latin` \| `devanagari` \| `arabic` \| `mixed` \| ... — matters for Romanized Hindi/Urdu. Derived. |
| `published_at` | datetime (UTC) \| null | When the content was published at source. |
| `edited_at` | datetime (UTC) \| null | Last edit at source, if known (Telegram edits, article updates). |
| `author_native_id` | string \| null | Source's author/channel identifier. |
| `author_display_name` | string \| null | Human-readable author/channel name. |
| `engagement` | JSON | Source-specific engagement blob (views, reactions, likes, comment counts). Schema per source in §2–4. |
| `media` | JSON array \| null | Attached media descriptors: type, source file id, content hash (text/pHash/video fp), caption. |
| `entities` | JSON | Extracted entities: urls[], hashtags[], mentions[], handles[], named_entities[]. |
| `source_specific` | JSON | All fields unique to the source that don't fit the core. Populated per §2–4. |
| `raw_payload_ref` | string | Pointer to stored raw payload (path/key). Raw is stored separately, not inline. |
| `provenance` | JSON | Collection provenance block — see §5. Mandatory, never null. |
| `extraction_confidence` | float (0–1) \| null | For unstructured sources: parser's confidence. Null for clean API sources (implicitly 1.0). |
| `content_hashes` | JSON | `{text_hash, phash, video_fp, url_hashes[]}` — for cross-item / cross-source matching. Derived. |

Two additional fields, `account_created_at` and derived `account_age_at_observation`, exist on the model but are populated only for promoted author/channel nodes — see §9.

---

## 2. Telegram — `source_specific` fields

Telegram is the **primary coordination source**. Forwarding chains are the richest structure we collect.

**Content / message**
- `message_id`, `channel_id`, `channel_username`, `media_type` (photo/video/document/none), `media_file_id`, `media_file_hash`, `caption`, `has_scheduled_flag`, `is_edited`, `deletion_checked_at`, `is_deleted`.

**Channel-level (captured per message, also as standalone channel items)**
- `channel_title`, `channel_description`, `channel_created_at`, `channel_verified`, `subscriber_count`, `channel_type` (broadcast/group), `linked_chat_id`, `is_public`.

**Forwarding (the coordination goldmine → generates EDGES, see §6)**
- `forward_from_channel_id`, `forward_from_message_id`, `forward_origin_timestamp`, `forward_latency_seconds` (derived: this send ts − origin ts), `forward_chain_depth`.

**Engagement blob**
- `{view_count, reactions: {emoji: count}, reply_count, forward_count, comment_thread_present}`.

**Threading → generates EDGES**
- `reply_to_message_id`.

---

## 3. YouTube — `source_specific` fields

Two `source_type`s: `youtube_video` and `youtube_comment`. Video/narrative dimension + comment-amplification sub-graph.

**Video-level (`youtube_video`)**
- `video_id`, `channel_id`, `title`, `description`, `tags[]`, `category_id`, `default_language`, `caption_available`, `duration_seconds`, `definition`, `is_live_broadcast`, `thumbnail_urls`.
- Channel: `channel_title`, `channel_description`, `channel_custom_url`, `channel_created_at`, `channel_country_declared` (weak coarse-origin cue), `subscriber_count`, `channel_total_views`, `channel_video_count`, `uploads_playlist_id`, `channel_keywords`.
- Engagement: `{view_count, like_count, comment_count, favorite_count}`.

**Comment-level (`youtube_comment`)**
- `comment_id`, `video_id` (→ `parent_item_id`), `author_channel_id`, `author_display_name`, `like_count`, `parent_comment_id` (reply structure → EDGE), `total_reply_count`.

**Network → generates EDGES**
- Shared descriptions/tags across channels, shared external URLs in descriptions, playlist co-membership, cross-channel commenting.

---

## 4. Fact-check + News — `source_specific` fields

Unstructured HTML. **Two roles, distinguished by `source_type`:**
- `factcheck` — **Role A: labels + origin hints.** Source-asserted verdicts; captured as claims, NOT system truth.
- `news` — **Role B: organic baseline + narrative context.** Primarily the negative/organic class.

**Article-level (both)**
- `canonical_url`, `headline`, `body_text`, `byline`, `publication_name`, `section`, `summary`, `updated_at`.

**Structured metadata when present (parse opportunistically)**
- Open Graph: `og_title`, `og_description`, `og_image`, `og_type`, `og_url`.
- Twitter Card tags.
- Schema.org JSON-LD: `NewsArticle`, and critically **`ClaimReview`** for fact-checks.
- `meta_description`, `meta_keywords`, `canonical_link`, `hreflang[]`.

**Fact-check-specific (`factcheck`, Role A — from `ClaimReview` where available)**
- `claim_reviewed` (the claim text), `verdict_rating` (false/misleading/true/…), `claimed_origin` (where the misinfo reportedly came from — feeds Tier-2 origin inference downstream), `entities_named[]`, `debunked_source_urls[]`, `claim_first_seen_at`, `checked_at`.
- **Captured as source assertion.** A field `asserting_source` records *who* made the verdict. The system does not treat a fact-check verdict as ground truth; it records that a named fact-checker asserted it.

**Links / embeds → generates cross-source EDGES (the payoff)**
- `outbound_links[]`, `embedded_social[]` (quoted tweets, embedded Telegram/YouTube — each embed is a hard edge to that specific item), `citations[]`.

**Provenance (heavier here — see §5)**
- `fetch_method` (rss/html_parse), `robots_txt_compliant` (bool), `source_terms_ref`, `raw_html_ref`, `parser_version`.

---

## 5. Provenance block (mandatory, every item)

Records collection facts, kept strictly separate from content facts.

| Field | Type | Notes |
|---|---|---|
| `collector_id` | string | Which collector produced this (e.g. `telegram_v1`, `youtube_v1`, `web_v1`). |
| `collected_at` | datetime (UTC) | When we fetched it — distinct from `published_at`. |
| `source_api_version` | string \| null | API/endpoint version. |
| `fetch_method` | enum | `api` \| `rss` \| `html_parse`. |
| `http_status` | int \| null | For web fetches. |
| `robots_txt_compliant` | bool \| null | For web fetches — records we respected robots.txt. |
| `source_terms_ref` | string \| null | Pointer to captured terms/ToS at fetch time. |
| `raw_payload_ref` | string | Where the raw JSON/HTML is stored. |
| `parser_version` | string \| null | For extracted sources. |
| `collection_run_id` | string | Batch/run this belongs to (reproducibility). |

---

## 6. `edge` model (the coordination graph)

Edges are captured at collection where visible, and computed later where derived. Each edge is one row.

| Field | Type | Notes |
|---|---|---|
| `edge_id` | string (UUID) | Primary key. |
| `edge_type` | enum | See below. |
| `src_item_id` | string (UUID) | Source node. |
| `dst_item_id` | string (UUID) \| null | Destination node (null if dst is external/unresolved). |
| `dst_external_ref` | string \| null | For edges to not-yet-collected targets (a URL we haven't fetched). |
| `directed` | bool | Most edges directed. |
| `weight` | float \| null | For weighted edges (e.g. shared-content similarity). |
| `observed_at` | datetime (UTC) | When the relationship was observed. |
| `origin` | enum | `collected` (seen at collection) \| `derived` (computed later). |
| `evidence` | JSON | What supports this edge (shared hash value, forward metadata, embed html). |

**Edge types:**
- `forward` — Telegram forward (src forwards dst). *Collected.* The core coordination signal.
- `reply` — Telegram reply / YouTube comment reply. *Collected.*
- `mention` — src mentions dst channel/handle. *Collected.*
- `embed` — news/factcheck article embeds/links a specific item. *Collected.* Cross-source.
- `debunks` — factcheck item targets a specific content item. *Collected/derived.* Cross-source, Role A.
- `shared_media` — two items share a media content hash. *Derived.* Strong coordination signal.
- `shared_url` — two items share a normalized outbound URL. *Derived.*
- `near_duplicate_text` — two items share near-identical normalized text (templating). *Derived.* Strong coordination signal.
- `co_author` — same author across items/platforms. *Derived.*
- `temporal_cocluster` — items published in a tight window around the same narrative. *Derived.*

---

## 7. What this schema deliberately does NOT do

- **No coordination verdict.** "Is this a coordinated network" is a downstream (detection + graph) decision. Collection provides edges and signals; it does not conclude.
- **No origin/actor attribution.** Coarse-origin *cues* are captured as fields (declared country, language/script, fact-check claimed-origin) but not resolved into a verdict here.
- **No treating fact-check verdicts as truth.** They are source assertions with a named asserter.
- **No labels baked into items.** Ground-truth labelling (for the eventual model work) is a separate annotation layer keyed on `item_id`, not a collection field.
- **No narrative/campaign scoring.** Coordination-network / campaign-level scoring is the detection layer's job — a separate spec. Collection captures the graph; it does not score it (design principle 5, §0 — also see §9's derived-trajectory note).
- **WhatsApp — permanent blind spot, not deferred.** End-to-end encrypted; no lawful bulk-OSINT collection path exists. Excluded by design, not by current capability.
- **Facebook — deferred, not excluded.** The schema fits Facebook content without modification (it's another `item` source). The blocker is sourcing/access (API restrictions), not schema design. Revisit if/when a collection path becomes available.

---

## 8. Resolved decisions (locked)

1. **Storage / persistence pipeline — LOCKED.** `raw (JSON/parquet on disk) → DuckDB (landing, processing, edge derivation) → clean node + edge lists → graph DB (deferred)`. DuckDB is the dev/processing engine (dedup, normalization, hash + derived-edge computation — all set-oriented analytical work). The graph DB is the serving/traversal layer, added when the coordination-topology UI needs multi-hop traversal / community detection / centrality. **The edge list (flat `(src, dst, type, weight, evidence)`) is the interface between them** — the native import format for any graph DB. **`item_id` UUIDs are carried through as graph node keys** — the graph DB does not mint its own IDs. DuckDB is deliberately NOT used as a graph engine (no native multi-hop traversal); that is the graph DB's job. Graph DB choice deferred (candidates: KùzuDB — embedded, DuckDB-adjacent, reads parquet; or Neo4j — heavyweight standard with mature viz tooling).

2. **Channel/author as nodes — LOCKED (yes).** `channel` and `author` are promoted to first-class item/node types, not attributes riding along on messages. Author-level and channel-level edges (cross-platform same-author, channel→channel mention graphs) reference these nodes directly.

3. **Media storage — hashes + refs only** (store content hashes and source refs, not the media files themselves at MVP; fetch on demand). Storage + legal simplicity.

4. **Dedup identity — LOCKED (two items + edge).** The same content appearing in two places is preserved as two distinct nodes joined by a `near_duplicate` (or `shared_media`) edge — never collapsed into one item. Rationale: the fact that it appeared in two places *is* the coordination signal.

5. **Fact-check verdicts — LOCKED (source assertions).** Captured as named assertions (`asserting_source` + verdict), never as system ground truth. See §4.

6. **Extraction confidence scale.** Single float for now; revisit per-field only if a downstream layer needs it.

---

## 9. Author/channel nodes — observation history + reputation fields (LOCKED)

Promoted author/channel `item`s (§8.2) carry two kinds of data that must not be conflated:

- **Identity fields — stable, live on the node.** `account_created_at` (account/channel creation date, populated from the source only when it is a real source-declared value). A derived `account_age_at_observation` (creation date → this item's collection time) rides alongside it. **`account_created_at` is the core dummy-account detection signal** — a brand-new account posting at high volume is exactly the pattern this field exists to make computable. **Telegram is a platform limitation, not a bug:** the API exposes no creation-date field anywhere, so `account_created_at` (and therefore `account_age_at_observation`) stays `null` for every Telegram channel item, full stop — no approximation is written into the real field. A best-effort estimate (the channel's earliest retrievable message timestamp) is instead captured under `source_specific.estimated_creation_from_earliest_msg`, explicitly named as an estimate so it can never be mistaken for the source-declared value downstream.
- **Reputation fields — volatile, NEVER overwritten on the node.** Subscriber/follower count, post count, verified status, and view count change on every run. Writing them onto the node and overwriting on each pass would destroy the account's trajectory — exactly the signal a dummy/sockpuppet-account detector needs (e.g. brand-new account, near-zero followers, sudden high-volume posting).

**Observations are a time series, not a snapshot.** Every collection run appends one `observation` row per author/channel node it saw, to a *separate* output — `data/raw/observations/`, parquet, one file per `collection_run_id` — rather than mutating the node or an embedded list on it. Parquet files are immutable once written, so "append" is naturally "write a new file": there is no in-place mutation to get wrong. DuckDB reads the whole directory as one time-series table (`read_parquet('data/raw/observations/*.parquet')`) with no merge logic required on the collection side. Re-running the collector must always add a new file, keyed on a fresh `collection_run_id`; it must never open and rewrite a prior run's file.

**`observation` fields:**

| Field | Type | Notes |
|---|---|---|
| `node_item_id` | string (UUID) | The author/channel `item_id` this observation is about. |
| `observed_at` | datetime (UTC) | When this observation was taken (= this run's collection time). |
| `subscriber_or_follower_count` | int \| null | |
| `view_count` | int \| null | Lifetime channel views, where the source exposes it (e.g. YouTube `statistics.viewCount`). Null for sources that don't (Telegram has no channel-level view total). |
| `post_count_seen` | int \| null | Posts/messages seen by this collector as of this run — not necessarily the account's lifetime total. YouTube's `statistics.videoCount` (a real lifetime total from the source) is mapped here as the closest available field. |
| `verified_status` | bool \| null | |
| `collection_run_id` | string | Ties the observation to the run/batch that produced it (§5). |

**Downstream, not here.** Follower-growth rate, posting-velocity change, and other trajectory metrics are *derived* from the `observations` time series by the DuckDB processing layer, not computed at collection time (design principle 5, §0 — collection captures signal, it does not interpret it).
