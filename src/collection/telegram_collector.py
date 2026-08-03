"""Telegram collector.

Will authenticate against the Telegram MTProto API via Telethon (using the
credentials loaded in config.py) and pull channel history — messages, replies,
forwards, and engagement metadata — normalizing each message into the
canonical `Item`/`Edge` schema defined in schema.py. No logic implemented yet.
"""
