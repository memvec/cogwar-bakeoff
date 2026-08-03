"""Output writers for collected data.

Will write canonical `Item` records as JSON (one file or JSON-lines per
collection run) and `Edge` records as parquet or JSONL, landing everything
under the configured raw data output path (see config.py, default
data/raw/). No logic implemented yet.
"""
