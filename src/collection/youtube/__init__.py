"""YouTube source collector. Maps YouTube Data API v3 objects into the
shared canonical schema (collection.schema) and writes via the shared
writers (collection.writers) -- same shape as collection.telegram. All
YouTube-specific code lives in this subpackage only: mapping.py holds the
API-object -> Item/Edge/Observation translation, collector.py orchestrates
(auth, search, batch-fetch, quota tracking, writing).
"""
