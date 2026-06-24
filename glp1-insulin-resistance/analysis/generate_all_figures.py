"""
Master figure-generation script: runs all three analyses in this project and
consolidates every output figure into ../figures/ with descriptive, report-ready
filenames, for direct insertion into a manuscript draft.

Runs:
  1. gse34451_hippocampus_analysis.py — real GEO transcriptomic analysis
     (Hippocampus, Control vs T2D)
  2. glp1_model.py mediation — calibrated illustration + power analysis
     (clearly-labeled, not real evidence — see that module's docstring)
  3. adni_pipeline.py --demo — synthetic schema-matched ADNI mediation demo
     (NOT real ADNI data — real mode requires your own approved access)

Usage (Colab or local):
    pip install -r requirements.txt
    python generate_all_figures.py

All consolidated figures land in ../figures/, prefixed by source:
    geo_*.png       (GSE34451 analysis)
    mediation_*.png (illustration + power curve)
    adni_*.png      (ADNI demo)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ANALYSIS_DIR.parent / "figures"


def run(cmd: list[str]) -> None:
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}\n{'='*70}")
    subprocess.run(cmd, check=True, cwd=ANALYSIS_DIR)


def collect(src_dir: Path, prefix: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        print(f"WARNING: expected output dir {src_dir} not found; skipping collection for prefix '{prefix}'")
        return
    for png in src_dir.glob("*.png"):
        dest = FIGURES_DIR / f"{prefix}_{png.name}"
        shutil.copy2(png, dest)
        print(f"  -> {dest.relative_to(ANALYSIS_DIR.parent)}")


def main() -> None:
    py = sys.executable

    # 1. Real GEO transcriptomic analysis (GSE34451)
    run([py, "gse34451_hippocampus_analysis.py"])
    collect(ANALYSIS_DIR / "output" / "GSE34451_hippocampus_t2d", "geo")

    # 2. Mediation model: illustration + power analysis
    run([py, "glp1_model.py", "mediation", "--simulate-illustration",
         "--outdir", "output/mediation_illustration"])
    collect(ANALYSIS_DIR / "output" / "mediation_illustration", "mediation_illustration")

    run([py, "glp1_model.py", "mediation", "--power-analysis",
         "--true-a", "0.27", "--true-b", "0.5", "--true-cprime", "1.75",
         "--n-list", "62,150,300,600,1200,3808", "--n-sims", "300",
         "--outdir", "output/power"])
    collect(ANALYSIS_DIR / "output" / "power", "mediation_power")

    # 3. ADNI demo (synthetic — real mode requires your own ADNI access)
    run([py, "adni_pipeline.py", "--demo", "--outcome", "adas13", "--outdir", "output/adni_demo"])
    collect(ANALYSIS_DIR / "output" / "adni_demo", "adni_demo")

    print(f"\nAll figures consolidated in: {FIGURES_DIR.resolve()}")
    print("Files:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
