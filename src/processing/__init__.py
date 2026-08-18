"""Processing layer: loads raw collection output (src/collection/) into a
clean, queryable DuckDB database and derives coordination edges from it.
Raw files under data/raw/ remain the source of truth; the DuckDB database
under data/processed/ is fully derived and disposable -- rebuilt from
scratch on every run (see build.py).
"""
