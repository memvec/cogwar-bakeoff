# Architectural Decision Log

This is the durable record of cross-cutting decisions for the venture — the ones that would otherwise live only in conversation and get re-litigated. Dated list, one-line decision + short rationale each. Update this as new decisions are made; don't let it drift out of sync with what's actually built.

---

## 2026-08-19

**Model/provider is a commodity, swappable at config level.**
All model-backed work (classifier, entity extraction, stance, transcripts) sits behind swappable interfaces (`EntityExtractor`, `TranscriptProvider`, and their future siblings). Defer the Claude→Gemini cost switch until all analysis passes are built, then do one corpus-wide head-to-head on cost vs. quality. Caveat: Gemini's free tier may train on submitted data — use paid tier for real collected content (sovereignty).

**The moat is the accumulated data + graph over time, not the model.**
Coordination topology and observation history compound and are defensible; models are replaceable. Build priorities favor collection breadth/continuity and graph quality over model performance.

**API-first for all model work, behind swappable interfaces.**
Local/self-hosted models deferred; not needed for dev or demo. Revisit at production scale.

**Data sovereignty is a parked decision, flagged not answered.**
Where collected content + the accumulated graph live, and what external services touch them, must be decided with Pushpahas + the government customer's constraints before the real product handles real intelligence data. For dev on benign/OSINT data, external APIs are acceptable.

**Analysis surfaces evidence + confidence, never verdicts.**
Consistency (author property) and coordination (cross-author synchrony) stay strictly separate — the false-positive guard. Every finding is an evidence package for the future Response layer.

**Collection: two-items-plus-edge dedup; observations are timestamped history, not snapshots; WhatsApp/Facebook.**
Duplicate content is preserved as two items joined by an edge, never collapsed. Author/channel observations (follower counts, etc.) are appended as a time series, never overwritten in place. WhatsApp is a permanent blind spot (E2E encrypted, no lawful bulk collection path); Facebook is deferred (access, not schema, is the blocker).

**Persistence pipeline: raw → DuckDB (processing/derivation) → graph DB (deferred, for traversal/UI).**
The edge list is the interface between layers; `item_id` UUIDs are the graph node keys, carried through unchanged. Graph DB choice deferred until the coordination-topology UI needs multi-hop traversal.
