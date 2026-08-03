"""Pydantic models for the canonical `Item` and `Edge` schema.

Implements the schema defined in docs/collection_schema.md: every collected
object (Telegram message, YouTube video/comment, news article, fact-check
article) normalizes into a single `Item` model; every relationship between
items (reply, forward, embed, etc.) is a separate `Edge` model. No models are
defined yet — this module is a placeholder for that implementation.
"""
