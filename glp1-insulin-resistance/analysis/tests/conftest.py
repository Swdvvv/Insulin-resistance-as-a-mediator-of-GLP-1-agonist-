"""Ensure the parent `analysis/` directory (where glp1_model.py and adni_pipeline.py
live) is importable regardless of which directory pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
