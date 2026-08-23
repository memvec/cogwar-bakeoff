"""serve package settings -- just the two DB paths.

Deliberately does NOT import analysis.config: that module fails loudly at
import time if ANTHROPIC_API_KEY is unset (by design, for the LLM-backed
passes), and this is a read-only viz backend with no LLM calls at all --
forcing API-key setup just to read two constant paths would violate the
same "don't force unrelated secrets" reasoning analysis.config itself uses
for staying independent of collection.config. The paths are duplicated
(not imported) for that reason -- a small, deliberate redundancy.
"""

from pathlib import Path

PROCESSED_DB_PATH = Path("data/processed/cogwar.duckdb")
ANALYSIS_DB_PATH = Path("data/analysis/analysis.duckdb")
