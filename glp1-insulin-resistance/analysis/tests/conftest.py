"""Ensure the parent `analysis/` directory (where geo_de_pathway_pipeline.py lives)
is importable regardless of which directory pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
