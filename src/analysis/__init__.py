"""Analysis layer (docs/analysis_layer_spec.md): consumes the processed
DuckDB (src/processing/) and produces entity/stance/narrative/finding
structures in a dedicated analysis DuckDB.

Only the first pass -- entity extraction + resolution -- is implemented so
far (entities.py, resolution.py, extract_entities.py). Stance detection,
narrative clustering, profile aggregation, and finding assembly are later
passes per the spec's §4 pipeline and are deliberately not built yet.
"""
