"""Processing layer paths.

Deliberately independent of collection.config: processing only reads
already-collected raw files and writes a DuckDB database, so it needs no
API credentials. Importing collection.config here would wrongly force every
processing run to have Telegram/YouTube secrets configured.
"""

from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
ITEMS_DIR = RAW_DATA_PATH / "items"
EDGES_DIR = RAW_DATA_PATH / "edges"
OBSERVATIONS_DIR = RAW_DATA_PATH / "observations"

PROCESSED_DATA_PATH = Path("data/processed")
DB_PATH = PROCESSED_DATA_PATH / "cogwar.duckdb"
