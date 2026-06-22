"""
ADNI secondary-analysis mediation pipeline for testing:
    GLP-1RA medication use (X) --a--> peripheral insulin resistance, HOMA-IR (M) --b--> cognitive/MRI decline (Y)

REQUIRES YOUR OWN ADNI DATA ACCESS. This script does not and cannot download
ADNI data. Apply at https://adni.loni.usc.edu (free, ~1-2 day approval), then
download these tables from the LONI Image and Data Archive (IDA):
    - ADNIMERGE.csv      (demographics, diagnosis, cognition, MRI volumes)
    - RECCMEDS.csv        (recent/concomitant medication log)
    - a lab/biomarker table containing fasting glucose AND fasting insulin
      (core ADNIMERGE has glucose-adjacent fields but NOT insulin in most
      phases; insulin is typically in an ancillary biomarker/metabolomics
      panel. Search the IDA Data Dictionary for "insulin" to find the exact
      file for your ADNI phase/download — the filename varies by cohort
      version, so it cannot be hardcoded here.)

WHAT THIS IS AND ISN'T:
  - This is an OBSERVATIONAL secondary-data analysis, not a trial. GLP-1RA
    "exposure" here means a patient happened to be on the drug (almost always
    for diabetes), not random assignment. Any mediation result describes an
    association consistent with the hypothesis, not proof of causal mediation.
  - HOMA-IR is a PERIPHERAL (blood) insulin-resistance proxy, not a measure of
    CENTRAL (brain) insulin resistance, which is what the research question
    actually asks about. Treat this as the closest available real-data proxy,
    not as a direct test of the central-IR hypothesis.
  - ADNI is an Alzheimer's-focused cohort; it has no Parkinson's disease arm
    (see PPMI for that, not covered by this script).

Usage:
    # Real ADNI data, once you have it:
    python adni_pipeline.py --adnimerge ADNIMERGE.csv --medications RECCMEDS.csv \
        --biomarkers insulin_glucose.csv --outcome adas13 --outdir output/adni

    # Synthetic schema-matched demo (NOT real data) to verify the pipeline runs:
    python adni_pipeline.py --demo --outcome adas13 --outdir output/adni_demo
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import mediation_model as mm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("adni_pipeline")

GLP1RA_NAME_PATTERN = re.compile(
    r"exenatide|byetta|bydureon|liraglutide|victoza|saxenda|"
    r"semaglutide|ozempic|wegovy|rybelsus|"
    r"dulaglutide|trulicity|lixisenatide|adlyxin|albiglutide|tanzeum",
    flags=re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# HOMA-IR
# --------------------------------------------------------------------------- #

def compute_homa_ir(glucose_mg_dl: pd.Series, insulin_uU_ml: pd.Series) -> pd.Series:
    """Standard HOMA-IR = (fasting glucose [mg/dL] * fasting insulin [uU/mL]) / 405."""
    return (glucose_mg_dl * insulin_uU_ml) / 405.0


# --------------------------------------------------------------------------- #
# LOADERS
# --------------------------------------------------------------------------- #

def load_adnimerge(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    keep = ["RID", "VISCODE", "EXAMDATE", "AGE", "PTGENDER", "PTEDUCAT",
            "APOE4", "DX", "ADAS13", "MMSE", "Hippocampus", "ICV"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        log.warning("ADNIMERGE missing expected columns: %s (continuing with what's present)", missing)
    return df[[c for c in keep if c in df.columns]].copy()


def flag_glp1ra_users(medications: pd.DataFrame, rid_col: str = "RID",
                       med_name_col: str = "CMMED") -> pd.DataFrame:
    """Returns one row per RID with a boolean glp1ra_user flag, based on a
    regex match against known GLP-1RA brand/generic names in the medication log."""
    is_match = medications[med_name_col].astype(str).str.contains(GLP1RA_NAME_PATTERN, na=False)
    flagged = (
        medications.assign(glp1ra_user=is_match)
        .groupby(rid_col)["glp1ra_user"].any()
        .reset_index()
    )
    return flagged


def load_biomarkers(path: str | Path, rid_col: str = "RID",
                     glucose_col: str = "GLUCOSE", insulin_col: str = "INSULIN") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in (rid_col, glucose_col, insulin_col):
        if c not in df.columns:
            raise KeyError(
                f"Column '{c}' not found in biomarker file. ADNI ancillary biomarker "
                f"files vary in column naming across panels/phases — open the file and "
                f"pass the correct --glucose-col/--insulin-col names."
            )
    df["homa_ir"] = compute_homa_ir(df[glucose_col], df[insulin_col])
    return df[[rid_col, "homa_ir"]].groupby(rid_col).mean().reset_index()


# --------------------------------------------------------------------------- #
# COHORT ASSEMBLY
# --------------------------------------------------------------------------- #

def build_cohort(adnimerge: pd.DataFrame, glp1ra_flags: pd.DataFrame,
                  homa_ir: pd.DataFrame, outcome: str = "adas13") -> pd.DataFrame:
    """
    Builds one row per RID with:
      treatment = glp1ra_user (0/1)
      mediator  = homa_ir (baseline)
      outcome   = change in ADAS13 (cognition, higher = worse) or normalized
                  hippocampal volume (Hippocampus / ICV, higher = better),
                  computed as last-available-visit minus baseline-visit.
      covariates = age, sex (numeric), education, apoe4 carrier status

    Requires at least two visits per subject in adnimerge to compute a change
    score; subjects with only one visit are dropped (logged).
    """
    outcome_col_map = {"adas13": "ADAS13", "hippocampus": "Hippocampus"}
    if outcome not in outcome_col_map:
        raise ValueError(f"outcome must be one of {list(outcome_col_map)}, got {outcome!r}")
    raw_col = outcome_col_map[outcome]

    df = adnimerge.dropna(subset=["EXAMDATE"]).copy()
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"])
    df = df.sort_values(["RID", "EXAMDATE"])

    n_before = df["RID"].nunique()
    visit_counts = df.groupby("RID").size()
    multi_visit_rids = visit_counts[visit_counts >= 2].index
    dropped = n_before - len(multi_visit_rids)
    if dropped:
        log.info("Dropping %d/%d subjects with fewer than 2 visits (can't compute change score)",
                  dropped, n_before)
    df = df[df["RID"].isin(multi_visit_rids)]

    baseline = df.groupby("RID").first()
    last = df.groupby("RID").last()

    if outcome == "hippocampus":
        baseline_val = baseline["Hippocampus"] / baseline["ICV"]
        last_val = last["Hippocampus"] / last["ICV"]
    else:
        baseline_val = baseline[raw_col]
        last_val = last[raw_col]

    cohort = pd.DataFrame({
        "RID": baseline.index,
        "age": baseline["AGE"],
        "sex_male": (baseline["PTGENDER"] == "Male").astype(float) if "PTGENDER" in baseline else np.nan,
        "education": baseline.get("PTEDUCAT", np.nan),
        "apoe4_carrier": (baseline.get("APOE4", np.nan) > 0).astype(float) if "APOE4" in baseline else np.nan,
        "outcome_change": (last_val - baseline_val).values,
    }).reset_index(drop=True)

    cohort = cohort.merge(glp1ra_flags.rename(columns={"glp1ra_user": "treatment"}), on="RID", how="left")
    cohort["treatment"] = cohort["treatment"].fillna(False).astype(float)
    cohort = cohort.merge(homa_ir.rename(columns={"homa_ir": "mediator"}), on="RID", how="left")

    return cohort


# --------------------------------------------------------------------------- #
# SYNTHETIC SCHEMA-MATCHED DEMO (NOT REAL DATA)
# --------------------------------------------------------------------------- #

def generate_synthetic_adni_demo(n: int = 400, true_a: float = -0.8, true_b: float = 0.3,
                                  true_cprime: float = -0.5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates THREE fake tables with ADNI-like column names/structure (NOT real
    ADNI data, NOT calibrated to any real published ADNI statistic) purely so
    the full load -> merge -> mediate -> plot pipeline can be exercised end to
    end before real ADNI access is available. true_a/b/cprime let you inject a
    known mediation structure to confirm the pipeline recovers it.
    """
    rng = np.random.default_rng(seed)
    rids = np.arange(1, n + 1)
    treatment = rng.binomial(1, 0.15, size=n).astype(float)  # ~15% on GLP-1RA, realistic for a diabetic subgroup

    homa_ir = 2.5 + true_a * treatment + rng.normal(0, 1.0, size=n)
    homa_ir = np.clip(homa_ir, 0.1, None)
    outcome_change = true_cprime * treatment + true_b * homa_ir + rng.normal(0, 1.5, size=n)

    age = rng.normal(74, 7, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    educ = rng.normal(15, 3, size=n)
    apoe4 = rng.binomial(1, 0.35, size=n)
    baseline_adas13 = rng.normal(18, 8, size=n)

    adnimerge_rows = []
    for i, rid in enumerate(rids):
        for visit, dt in enumerate(["2020-01-01", "2021-06-01"]):
            adnimerge_rows.append({
                "RID": rid, "VISCODE": f"v{visit}", "EXAMDATE": dt,
                "AGE": age[i], "PTGENDER": sex[i], "PTEDUCAT": educ[i],
                "APOE4": apoe4[i],
                "DX": "MCI",
                "ADAS13": baseline_adas13[i] + (outcome_change[i] if visit == 1 else 0),
                "MMSE": np.nan, "Hippocampus": np.nan, "ICV": np.nan,
            })
    adnimerge = pd.DataFrame(adnimerge_rows)

    med_names = np.where(treatment == 1,
                          rng.choice(["Liraglutide", "Semaglutide", "Dulaglutide"], size=n),
                          "Metformin")
    medications = pd.DataFrame({"RID": rids, "VISCODE": "v0", "CMMED": med_names})

    glucose = rng.normal(100, 15, size=n)
    insulin = homa_ir * 405.0 / glucose
    biomarkers = pd.DataFrame({"RID": rids, "GLUCOSE": glucose, "INSULIN": insulin})

    return adnimerge, medications, biomarkers


# --------------------------------------------------------------------------- #
# VISUALIZATION
# --------------------------------------------------------------------------- #

def plot_cohort_overview(cohort: pd.DataFrame, outcome_label: str, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    counts = cohort["treatment"].map({1.0: "GLP-1RA user", 0.0: "Non-user"}).value_counts()
    axes[0].bar(counts.index, counts.values, color=["#d95f02", "#1b9e77"])
    axes[0].set_title("Cohort sizes")
    axes[0].set_ylabel("n subjects")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    groups = [cohort.loc[cohort["treatment"] == 0, "mediator"].dropna(),
              cohort.loc[cohort["treatment"] == 1, "mediator"].dropna()]
    axes[1].boxplot(groups, tick_labels=["Non-user", "GLP-1RA user"])
    axes[1].set_title("HOMA-IR by group")
    axes[1].set_ylabel("HOMA-IR")

    colors = cohort["treatment"].map({1.0: "#d95f02", 0.0: "#1b9e77"})
    axes[2].scatter(cohort["mediator"], cohort["outcome_change"], c=colors, alpha=0.6, s=20)
    for label, color in [("Non-user", "#1b9e77"), ("GLP-1RA user", "#d95f02")]:
        sub = cohort[cohort["treatment"] == (1.0 if label == "GLP-1RA user" else 0.0)].dropna(subset=["mediator", "outcome_change"])
        if len(sub) > 1:
            m, b = np.polyfit(sub["mediator"], sub["outcome_change"], 1)
            xs = np.linspace(sub["mediator"].min(), sub["mediator"].max(), 50)
            axes[2].plot(xs, m * xs + b, color=color, label=label)
    axes[2].set_xlabel("HOMA-IR (mediator)")
    axes[2].set_ylabel(outcome_label)
    axes[2].set_title("HOMA-IR vs outcome change")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved cohort overview to %s", outpath)


def plot_bootstrap_distribution(result: mm.MediationResult, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    sim = rng.normal(result.indirect_effect, max((result.indirect_ci[1] - result.indirect_ci[0]) / 4, 1e-6), 5000)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sim, bins=40, color="#7570b3", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(result.indirect_ci[0], color="red", linestyle=":", linewidth=1.2)
    ax.axvline(result.indirect_ci[1], color="red", linestyle=":", linewidth=1.2)
    ax.set_title("Approx. bootstrap distribution of indirect effect (a*b)")
    ax.set_xlabel("Indirect effect")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved bootstrap distribution plot to %s", outpath)


# --------------------------------------------------------------------------- #
# DRIVER
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adnimerge", help="Path to ADNIMERGE.csv")
    parser.add_argument("--medications", help="Path to RECCMEDS.csv (or equivalent medication log)")
    parser.add_argument("--biomarkers", help="Path to a lab/biomarker CSV containing glucose + insulin")
    parser.add_argument("--glucose-col", default="GLUCOSE")
    parser.add_argument("--insulin-col", default="INSULIN")
    parser.add_argument("--outcome", choices=["adas13", "hippocampus"], default="adas13")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="output/adni")
    parser.add_argument("--demo", action="store_true",
                         help="Use a synthetic, schema-matched demo dataset (NOT real ADNI data)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outcome_label = "Change in ADAS13 (higher = worse)" if args.outcome == "adas13" else "Change in Hippocampus/ICV (higher = better)"

    if args.demo:
        log.warning("=== DEMO MODE: synthetic schema-matched data, NOT real ADNI data ===")
        adnimerge, medications, biomarkers = generate_synthetic_adni_demo(seed=args.seed)
        glp1ra_flags = flag_glp1ra_users(medications)
        homa_ir = biomarkers.assign(homa_ir=compute_homa_ir(biomarkers["GLUCOSE"], biomarkers["INSULIN"]))[["RID", "homa_ir"]]
        cohort = build_cohort(adnimerge, glp1ra_flags, homa_ir, outcome="adas13")
    else:
        if not (args.adnimerge and args.medications and args.biomarkers):
            parser.error("--adnimerge, --medications, and --biomarkers are required unless --demo is set")
        adnimerge = load_adnimerge(args.adnimerge)
        medications = pd.read_csv(args.medications, low_memory=False)
        glp1ra_flags = flag_glp1ra_users(medications)
        homa_ir = load_biomarkers(args.biomarkers, glucose_col=args.glucose_col, insulin_col=args.insulin_col)
        cohort = build_cohort(adnimerge, glp1ra_flags, homa_ir, outcome=args.outcome)

    n_treated = int(cohort["treatment"].sum())
    log.info("Cohort assembled: n=%d total, %d GLP-1RA users, %d non-users",
              len(cohort), n_treated, len(cohort) - n_treated)
    if n_treated < 10:
        log.warning("Fewer than 10 GLP-1RA users in cohort — mediation estimates will be very unstable. "
                    "This is the expected, documented limitation of incidental/observational GLP-1RA exposure in ADNI.")

    cohort.to_csv(outdir / "cohort.csv", index=False)

    result = mm.fit_mediation(
        cohort, treatment="treatment", mediator="mediator", outcome="outcome_change",
        covariates=[c for c in ["age", "sex_male", "education", "apoe4_carrier"] if cohort[c].notna().all()],
        n_boot=args.n_boot, seed=args.seed,
        notes=["Observational/incidental GLP-1RA exposure in ADNI, not randomized — see module docstring.",
               "HOMA-IR is a peripheral proxy for insulin resistance, not a central/brain measure."],
    )
    print(result.summary())

    mm.plot_path_diagram(result, "GLP-1RA use", "HOMA-IR", outcome_label,
                          outdir / "adni_path_diagram.png", title="ADNI secondary analysis: mediation path model")
    plot_cohort_overview(cohort, outcome_label, outdir / "adni_cohort_overview.png")
    plot_bootstrap_distribution(result, outdir / "adni_bootstrap_distribution.png")


if __name__ == "__main__":
    main()
