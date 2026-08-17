"""Telegram source collector. Maps Telethon objects into the shared canonical
schema (collection.schema) and writes via the shared writers (collection.writers).
All Telegram-specific code lives in this subpackage only — mapping.py holds
the Telethon -> Item/Edge/Observation translation, collector.py orchestrates
(auth, iteration, rate-limiting, writing). Sibling subpackages
(collection.youtube, collection.news) will follow the same shape.
"""
