# Analysis Layer — Entity-Stance & Coordination Scoring Specification

**Status:** DRAFT v0.1 — for review
**Package:** `src/analysis/` (sibling to `collection`, `processing`)
**Consumes:** the processed DuckDB (`items`, `edges`, `nodes`, `observations`) + collected content.
**Produces:** entity nodes, entity-stance edges, author entity-stance profiles, narrative/topic clusters, coordination findings — all evidence-backed and queryable.
**Downstream consumer (not built yet):** the Response layer will package analysis *findings* into governed warnings. Therefore every finding must carry its supporting evidence, not just a score.

---

## 0. Design principles

1. **Surface with evidence + confidence; never assert verdicts.** Analysis outputs coordination *strength* and stance *signals* with confidence and attached evidence. It does NOT declare intent or attribution. Verdicts are for humans / governed downstream layers. (Carries the collection layer's "surface, don't judge" upward.)
2. **The unit of signal is entity-stance, not narrative-participation.** The core fingerprint is an author's *persistent stance toward an entity* across many topics — not participation in one content cluster. A stance-consistent account is defined by consistency across narratives, not by one campaign.
3. **Narrative and stance are orthogonal.** An item is *about* a topic/narrative while expressing *stance* toward an entity. The same narrative can carry opposite stances. Model them separately.
4. **Coordination = shared content + synchrony across DIFFERENT authors. Consistency = persistent stance, even with original wording.** These are different signals answering different questions (coordinated network vs. consistent independent agenda). Compute and display both; the contrast is the product.
5. **Every finding is an evidence package.** Any surfaced finding carries: the authors, entities, stance, coordination evidence (which shared content, what timing), confidence, and time range — structured so the future Response layer can cite it without re-deriving.
6. **Extraction is API-backed and swappable.** Entity extraction, entity resolution, and stance detection run via an LLM API behind a clean interface (same pattern as the classifier/transcript provider). Swappable for a local or different provider later.
7. **Longitudinal by default.** Profiles and scores accumulate across collection runs. A cross-narrative, multi-month stance trajectory is the moat-flavored output, not a single-batch snapshot.

---

## 1. Ontology (the backbone)

### 1.1 Entity
The thing stance is *about*.
- `entity_id` (canonical UUID), `canonical_name`, `entity_type` (enum: `country` / `person` / `brand` / `org` / `topic` / `other`), `aliases[]` (all surface forms resolved to this entity, including variant spellings, honorifics, script variants, and emoji/flag forms), `created_from` (seed vs discovered), `first_seen_at`, `observation_count`.
- **Hybrid population (locked):** a controlled seed list of high-priority entities is always tracked; open extraction surfaces *emerging* entities from the data for review and promotion into the tracked set. The seed list is derived from the venture's current collection scope / threat model and maintained in config (`configs/`), NOT hardcoded in this spec or in code.

### 1.2 Entity resolution (MUST — locked as v1 requirement)
Surface mentions in any language/script resolve to one canonical `entity_id` via the LLM API. Without this, a single real-world entity fractures across its many surface forms and the signal dissolves. Resolution is API-driven (the model canonicalizes variant forms to one entity), with a persisted alias→entity map so repeated mentions are cheap and consistent. New aliases get resolved once and cached.

### 1.3 Narrative / Topic
The subject matter an item is about.
- `narrative_id`, `label`, `topic_type` (event / theme), `member_items[]`, `time_range`, `population_basis` (see below: tight vs loose).
- **Two flavors, both computed (locked "we need both"):**
  - **Tight narrative (content cluster):** connected components of `near_duplicate_text` + `shared_media` edges. Catches copy-paste coordination. Buildable on existing data.
  - **Loose narrative (semantic cluster):** items grouped by embedding similarity (API embeddings), catching paraphrased/thematically-aligned content that isn't identical. Catches the consistent-agenda account that never copy-pastes.
  - **The difference between them is a first-class output:** tight-but-not-loose = pure copy-paste network; loose-but-not-tight = consistent independent messaging; both = coordinated network pushing a coherent narrative. This contrast is the coordinated-vs-genuine distinction one layer up.

### 1.4 Stance
An author's polarity toward an entity in a given item.
- Per (item, entity): `polarity` (positive / negative / neutral), `strength` (0–1), `confidence` (0–1). API-detected. Handles sarcasm / code-mixed text as best-effort with confidence (not ground truth — locked).

---

## 2. Core derived structures

### 2.1 Entity-stance edge (item → entity)
For each item, for each entity it references: an edge `item --[stance]--> entity` carrying polarity, strength, confidence, extraction provenance (which API/model/version). This is the atomic signal everything aggregates from.

### 2.2 Author entity-stance profile (the core insight)
Per author, aggregated across ALL their content and ALL narratives, per entity:
- `author_id`, `entity_id`, `net_stance` (aggregate polarity, e.g. strength-weighted mean), `stance_consistency` (how consistently pos/neg vs. mixed — a wobbly account scores low consistency), `volume` (how much they talk about it), `time_span`, `first_seen`/`last_seen`, `narrative_spread` (across how many distinct narratives they carried this stance — high spread = persistent agenda, not one campaign).
- **This answers "is this source persistently positive/negative toward entity X."** The persistent, cross-narrative stance vector per author.

### 2.3 Generate vs. amplify (author property)
Per author, orthogonal to stance: are they originating or echoing?
- Derived from forward direction (Telegram), who-posts-first within shared-content clusters (timing), ratio of original vs. forwarded/duplicate content.
- `origination_score` (0 = pure amplifier, 1 = pure originator), evidence = the timing/forward data.

### 2.4 Coordination finding (the evidence package)
When multiple authors show shared content + synchrony (temporal_cocluster) on a narrative, emit a *finding*:
- `finding_id`, `authors[]`, `narrative_id`, `entities[]` + dominant stance, `coordination_evidence` (shared hashes, time deltas, synchrony metrics), `confidence`, `time_range`, `generate_vs_amplify` roles of participants.
- **This is the Response-layer-ready unit.** Everything a future warning needs is attached.

---

## 3. The four output queries (ACCEPTANCE CRITERIA)

The layer is correct if it answers these four. Each is evidence-backed.

1. **Given an entity → topics currently discussed about it, over time, split positive/negative.**
   → Query entity-stance edges for that entity, group by narrative + time bucket, split by polarity. "What is being said about this entity, pro and anti, by topic, over time."

2. **Given a topic → positive/negative breakdown + coordination info.**
   → Take the narrative, show stance distribution (how much pro vs anti), and overlay coordination findings (which author clusters are synchronized on it). "Dissect one narrative."

3. **Given a subject → who talks positively/negatively consistently.**
   → Rank author entity-stance profiles for that entity by net_stance × consistency × narrative_spread. "Find the persistent agendas." (The core insight.)

4. **Given an author → their stance history across entities and narratives.**
   → The author's full profile: stance vector across entities, narratives participated in, generate/amplify role, coordination findings they appear in, over time. "Profile one actor."

---

## 4. Pipeline / where it runs

`raw + processed DuckDB → analysis passes → analysis tables (entities, entity_stance_edges, author_profiles, narratives, findings) in DuckDB → serves the four queries + feeds future graph UI + future Response layer.`

Passes (each swappable / re-runnable):
1. **Entity extraction + resolution** (API): per item → entities (resolved to canonical ids). Cache alias→entity.
2. **Stance detection** (API): per (item, entity) → polarity/strength/confidence.
3. **Narrative clustering:** tight (graph components, local) + loose (API embeddings + clustering).
4. **Profile aggregation** (local / DuckDB): roll up entity-stance edges → author profiles, longitudinal.
5. **Generate-vs-amplify** (local): timing/forward analysis → origination scores.
6. **Finding assembly** (local): coordination + stance + narrative → evidence-backed findings.

API passes are cost-bearing (per-item LLM calls) — must be incremental (only process new/unprocessed items, cache results keyed on item content hash) so re-runs are cheap. Reuses the checkpoint discipline from collection.

---

## 5. Open questions for review

1. **Cost control on API passes.** Entity+stance extraction is an LLM call per item; the current corpus is ~14k items. Batch multiple items per call? Sample rather than exhaustive? Process only items in coordination clusters first? Recommend: incremental + cache by content hash, batch where possible, and a priority/limit mode so a demo run is cheap. Needs a rough budget.
2. **Stance aggregation math.** How exactly net_stance + consistency combine (strength-weighted mean? entropy for consistency?). Defer exact formula to build, but the false-positive guard is a hard rule: a lone high-volume single-topic author must NOT score as coordinated — consistency and narrative_spread are *stance* signals; *coordination* requires cross-author synchrony (§2.4), kept strictly separate.
3. **Loose-narrative threshold.** Embedding similarity cutoff for semantic clusters — the tuning problem. Start with a conservative default, make it a param, validate against the tight clusters as a sanity check.
4. **Seed entity set.** The controlled seed list is maintained in config, derived from the current collection scope / threat model. Populated separately from this spec.
5. **Neutral/irrelevant entity handling.** Most items mention entities in passing with no real stance. Threshold below which we don't emit a stance edge (avoid noise)?
