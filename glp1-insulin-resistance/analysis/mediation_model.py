"""
Formal causal mediation analysis model for testing:
    GLP-1RA treatment (X) --a--> central insulin-resistance marker (M) --b--> neurodegenerative outcome (Y)
                          \\___________________ c' (direct effect) ____________________/

This is the literal analytical step this whole project has found missing from every
paper reviewed (see ../extraction/literature_extraction.json -> synthesis_summary.key_finding:
"No identified study ... performed a formal statistical mediation analysis").

WHAT THIS MODEL CAN AND CANNOT DO — read before using any output as "evidence":

1. No public individual-patient-data (IPD) dataset exists that pairs GLP-1RA
   treatment, a quantified central-insulin-resistance marker, and a
   neurodegenerative outcome in the SAME subjects. This was confirmed across
   this project's full literature search (21+ papers) and GEO transcriptomics
   search. Therefore this script CANNOT currently produce new evidence for or
   against Hypothesis 1 or Hypothesis 3 — the necessary input data simply isn't
   public.

2. What it DOES provide:
   a) fit_mediation() — a real, statistically correct Baron-Kenny path-model with
      bootstrapped indirect-effect confidence intervals. Point it at a real CSV
      of patient-level data (treatment, mediator, outcome columns) the moment
      one becomes available — e.g., if a trial sponsor or the Athauda/Foltynie
      group released IPD — and it will produce a genuine mediation estimate.
   b) simulate_athauda2019_illustration() — a CALIBRATED SIMULATION, not real
      data. It reconstructs plausible patient-level data whose treatment effect
      on the mediator matches the one real aggregate number this literature
      has (Athauda et al. 2019, JAMA Neurol: exenatide increased neuronal-
      exosome IRS-1 tyrosine phosphorylation by 0.27 AU, 95% CI 0.09-0.44,
      p=.003, at 48 weeks) and whose total treatment effect matches the parent
      trial's reported motor outcome (Athauda et al. 2017, Lancet: adjusted
      difference 3.5 points on MDS-UPDRS III, p=.0318). The mediator-to-outcome
      (b) path is NOT published with a quantified coefficient anywhere (the
      2019 paper reports only an F-statistic for mTOR's association with motor
      change, not a beta for IRS-1) — so b is an ASSUMED illustrative value,
      loudly flagged as such in every output. This mode answers "what WOULD a
      mediation analysis look like if you had IPD with these characteristics,"
      not "what mediation analysis was actually found."
   c) run_power_analysis() — given assumed true effect sizes, estimates whether
      trials of the sample sizes actually used in this literature (n=62, n=204,
      n=3808, etc.) are even large enough to detect mediation if it exists.
      This IS a legitimate, evidence-adjacent finding: it tells you whether the
      absence of a mediation analysis in the literature might partly reflect
      under-powered trials rather than (or in addition to) researchers simply
      not running the analysis.

Usage:
    # Real data, the moment it exists:
    python mediation_model.py --csv ipd.csv --treatment drug --mediator irs1_change --outcome updrs_change

    # Calibrated illustration (clearly labeled, not new evidence):
    python mediation_model.py --simulate-illustration

    # Power/feasibility analysis:
    python mediation_model.py --power-analysis --true-a 0.27 --true-b 0.5 --true-cprime 1.75 \
        --n-list 62,150,300,600,1200

    # Pure offline mechanism smoke-test (arbitrary params, no calibration claims):
    python mediation_model.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mediation_model")


# --------------------------------------------------------------------------- #
# CORE MEDIATION MODEL
# --------------------------------------------------------------------------- #

@dataclass
class MediationResult:
    n: int
    a_coef: float          # treatment -> mediator
    a_se: float
    a_pvalue: float
    b_coef: float          # mediator -> outcome (controlling for treatment)
    b_pvalue: float
    c_prime: float         # direct effect (treatment -> outcome, controlling for mediator)
    c_prime_pvalue: float
    total_effect: float    # treatment -> outcome, mediator not included (should ~= c_prime + a*b)
    indirect_effect: float  # a * b, the mediated ("explained by M") effect
    indirect_ci: tuple[float, float]  # bootstrap percentile CI for indirect_effect
    proportion_mediated: float        # indirect_effect / total_effect
    n_boot: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ci_excludes_zero = not (self.indirect_ci[0] <= 0 <= self.indirect_ci[1])
        lines = [
            f"Mediation analysis (n={self.n}, {self.n_boot} bootstrap resamples)",
            f"  Path a  (treatment -> mediator):           {self.a_coef:+.4f}  (p={self.a_pvalue:.4g} {sig(self.a_pvalue)})",
            f"  Path b  (mediator -> outcome | treatment): {self.b_coef:+.4f}  (p={self.b_pvalue:.4g} {sig(self.b_pvalue)})",
            f"  Direct effect c' (treatment -> outcome | mediator): {self.c_prime:+.4f}  (p={self.c_prime_pvalue:.4g} {sig(self.c_prime_pvalue)})",
            f"  Total effect c  (treatment -> outcome, mediator excluded): {self.total_effect:+.4f}",
            f"  Indirect (mediated) effect a*b: {self.indirect_effect:+.4f}",
            f"  95% bootstrap CI for indirect effect: [{self.indirect_ci[0]:+.4f}, {self.indirect_ci[1]:+.4f}]"
            f"  {'-> excludes zero (consistent with mediation)' if ci_excludes_zero else '-> includes zero (NOT statistically distinguishable from no mediation)'}",
            f"  Proportion of total effect mediated: {self.proportion_mediated:.1%}" if np.isfinite(self.proportion_mediated) else
            "  Proportion mediated: undefined (total effect ~= 0)",
        ]
        if self.notes:
            lines.append("  NOTES:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def fit_mediation(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    outcome: str,
    covariates: list[str] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
    notes: list[str] | None = None,
) -> MediationResult:
    """
    Baron-Kenny path-model mediation analysis with a nonparametric (case-resampling)
    bootstrap confidence interval for the indirect effect a*b. Treatment, mediator,
    and outcome must be numeric columns (binarize/dummy-code categoricals before
    calling this). Covariates, if given, must also be numeric.

    This is appropriate for continuous mediator + continuous outcome (linear OLS
    throughout). Binary outcomes require a different framework (e.g., counterfactual
    mediation a la Imai et al.) because the product-of-coefficients decomposition
    does not hold cleanly under nonlinear (e.g., logistic) outcome models — that is
    NOT implemented here; do not use this function on a binary/logistic outcome.
    """
    import statsmodels.api as sm

    covariates = covariates or []
    cols_needed = [treatment, mediator, outcome] + covariates
    data = df[cols_needed].dropna().reset_index(drop=True)
    n = len(data)
    if n < 10:
        raise ValueError(f"Only {n} complete-case rows available; mediation analysis needs more data.")

    def _fit_paths(d: pd.DataFrame) -> tuple[float, float, float, float, float, float, float, float]:
        X_a = sm.add_constant(d[[treatment] + covariates])
        model_a = sm.OLS(d[mediator], X_a).fit()
        a_coef = model_a.params[treatment]
        a_se = model_a.bse[treatment]
        a_p = model_a.pvalues[treatment]

        X_b = sm.add_constant(d[[treatment, mediator] + covariates])
        model_b = sm.OLS(d[outcome], X_b).fit()
        b_coef = model_b.params[mediator]
        b_p = model_b.pvalues[mediator]
        cprime = model_b.params[treatment]
        cprime_p = model_b.pvalues[treatment]

        X_total = sm.add_constant(d[[treatment] + covariates])
        model_total = sm.OLS(d[outcome], X_total).fit()
        total_effect = model_total.params[treatment]

        return a_coef, a_se, a_p, b_coef, b_p, cprime, cprime_p, total_effect  # type: ignore[return-value]

    a_coef, a_se, a_p, b_coef, b_p, cprime, cprime_p, total_effect = _fit_paths(data)
    indirect_effect = a_coef * b_coef

    rng = np.random.default_rng(seed)
    boot_indirect = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = data.iloc[idx]
        try:
            a_b, _, _, b_b, _, _, _, _ = _fit_paths(resampled)
            boot_indirect[i] = a_b * b_b
        except Exception:  # singular design matrix on a degenerate resample, etc.
            boot_indirect[i] = np.nan
    boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
    ci_lower, ci_upper = np.percentile(boot_indirect, [2.5, 97.5])

    proportion_mediated = indirect_effect / total_effect if abs(total_effect) > 1e-9 else float("nan")

    return MediationResult(
        n=n, a_coef=a_coef, a_se=a_se, a_pvalue=a_p, b_coef=b_coef, b_pvalue=b_p,
        c_prime=cprime, c_prime_pvalue=cprime_p, total_effect=total_effect,
        indirect_effect=indirect_effect, indirect_ci=(ci_lower, ci_upper),
        proportion_mediated=proportion_mediated, n_boot=len(boot_indirect),
        notes=notes or [],
    )


# --------------------------------------------------------------------------- #
# SIMULATION — generic (no calibration claims)
# --------------------------------------------------------------------------- #

def simulate_mediation_dataset(
    n: int = 200,
    true_a: float = 1.0,
    true_b: float = 1.0,
    true_cprime: float = 0.0,
    treatment_allocation: float = 0.5,
    mediator_noise_sd: float = 1.0,
    outcome_noise_sd: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic (treatment, mediator, outcome) data under a known,
    user-specified causal structure — purely a mechanism-testing tool, no
    relationship to any real trial unless the caller chooses parameters to match one."""
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, treatment_allocation, size=n).astype(float)
    mediator = true_a * treatment + rng.normal(0, mediator_noise_sd, size=n)
    outcome = true_cprime * treatment + true_b * mediator + rng.normal(0, outcome_noise_sd, size=n)
    return pd.DataFrame({"treatment": treatment, "mediator": mediator, "outcome": outcome})


# --------------------------------------------------------------------------- #
# SIMULATION — calibrated illustration of the Athauda et al. 2017/2019 scenario
# --------------------------------------------------------------------------- #

def simulate_athauda2019_illustration(
    assumed_proportion_mediated: float = 0.4, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Reconstructs a plausible patient-level dataset matching the AGGREGATE
    statistics actually published for the Exenatide-PD trial and its exosome
    sub-study, for the SOLE purpose of illustrating what a mediation analysis
    would look like with data of this shape. THIS IS NOT REAL PATIENT DATA.

    Calibration provenance (see returned 'notes' dict for full audit trail):
      - n=62 (32 exenatide / 30 placebo): REAL, from Athauda et al. 2017 (Lancet).
      - Path a (treatment -> mediator) target = 0.27 AU at 48 weeks, derived from
        Athauda et al. 2019 (JAMA Neurol): exosome IRS-1(pTyr) mean difference
        0.27 AU (95% CI 0.09-0.44, p=.003). REAL aggregate statistic.
      - Total effect target = 3.5 points on MDS-UPDRS III (off-medication), from
        Athauda et al. 2017: adjusted difference -3.5 (p=.0318), sign flipped here
        so higher = more improvement. REAL aggregate statistic.
      - Path b (mediator -> outcome) and the resulting split between direct and
        indirect effect: ASSUMED. The source papers report only an F-statistic
        for mTOR's association with motor outcome (F=5.343, p=.001), not a
        quantified IRS-1-specific regression coefficient, so there is no
        published number to calibrate b against. This function instead ASSUMES
        a user-specified proportion of the total effect is mediated
        (default 40%) and back-solves b to match — an arbitrary illustrative
        choice, not a finding. Re-run with different assumed_proportion_mediated
        values to see how sensitive the "result" is to this assumption (it is
        very sensitive, which is itself the point).
    """
    n_treat, n_control = 32, 30
    n = n_treat + n_control
    treatment_target_a = 0.27       # AU, real (Athauda 2019)
    total_effect_target = 3.5       # MDS-UPDRS III points, real (Athauda 2017), sign-flipped to "improvement"

    indirect_target = assumed_proportion_mediated * total_effect_target
    direct_target = total_effect_target - indirect_target
    b_assumed = indirect_target / treatment_target_a

    rng = np.random.default_rng(seed)
    treatment = np.array([1.0] * n_treat + [0.0] * n_control)
    rng.shuffle(treatment)

    # mediator noise SD backed out from the published 95% CI half-width (0.175 AU)
    # for the mean DIFFERENCE; approximate per-arm SD assuming roughly equal arms.
    diff_se = (0.44 - 0.09) / (2 * 1.96)
    mediator_sd = diff_se * np.sqrt(n_treat * n_control / n)
    mediator = treatment_target_a * treatment + rng.normal(0, mediator_sd, size=n)

    outcome_sd = 5.0  # illustrative only; MDS-UPDRS III off-med SD not published for this subgroup
    outcome = direct_target * treatment + b_assumed * mediator + rng.normal(0, outcome_sd, size=n)

    df = pd.DataFrame({"treatment": treatment, "mediator_irs1_pTyr_change": mediator,
                        "outcome_updrs3_improvement": outcome})

    notes = {
        "real_calibrated_values": {
            "n_treatment": n_treat, "n_control": n_control,
            "path_a_target_AU": treatment_target_a,
            "path_a_source": "Athauda et al. 2019, JAMA Neurol, PMC6459135",
            "total_effect_target_UPDRS_points": total_effect_target,
            "total_effect_source": "Athauda et al. 2017, Lancet, PMC5831666",
        },
        "assumed_illustrative_values": {
            "assumed_proportion_mediated": assumed_proportion_mediated,
            "back_solved_path_b": b_assumed,
            "outcome_noise_sd": outcome_sd,
            "warning": "Path b and the direct/indirect split are NOT published anywhere; "
                       "they are assumed for illustration only and the 'result' below is "
                       "entirely an artifact of that assumption, not a finding.",
        },
    }
    return df, notes


# --------------------------------------------------------------------------- #
# POWER / FEASIBILITY ANALYSIS
# --------------------------------------------------------------------------- #

def run_power_analysis(
    n_list: list[int],
    true_a: float,
    true_b: float,
    true_cprime: float,
    n_sims: int = 500,
    n_boot: int = 1000,
    mediator_noise_sd: float = 1.0,
    outcome_noise_sd: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each sample size in n_list, simulate n_sims independent trials under the
    given TRUE (assumed) effect sizes and compute the empirical power: the
    fraction of simulated trials whose 95% bootstrap CI for the indirect effect
    excludes zero. Answers: "if mediation of this size truly exists, how often
    would a trial of this size actually detect it?"
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_list:
        detections = 0
        for sim in range(n_sims):
            sim_seed = int(rng.integers(0, 2**31 - 1))
            df = simulate_mediation_dataset(
                n=n, true_a=true_a, true_b=true_b, true_cprime=true_cprime,
                mediator_noise_sd=mediator_noise_sd, outcome_noise_sd=outcome_noise_sd,
                seed=sim_seed,
            )
            result = fit_mediation(df, "treatment", "mediator", "outcome",
                                    n_boot=n_boot, seed=sim_seed)
            if not (result.indirect_ci[0] <= 0 <= result.indirect_ci[1]):
                detections += 1
        power = detections / n_sims
        rows.append({"n": n, "power": power, "n_sims": n_sims})
        log.info("n=%d: empirical power = %.2f (true_a=%.3g, true_b=%.3g)", n, power, true_a, true_b)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# VISUALIZATION
# --------------------------------------------------------------------------- #

def plot_path_diagram(result: MediationResult, treatment_label: str, mediator_label: str,
                       outcome_label: str, outpath: Path, title: str = "Mediation path model") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = {"X": (0.5, 1, 2.5, 1.2), "M": (4, 3.5, 2.5, 1.2), "Y": (7.5, 1, 2.5, 1.2)}
    labels = {"X": treatment_label, "M": mediator_label, "Y": outcome_label}
    for key, (x, y, w, h) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, labels[key], ha="center", va="center", fontsize=9, wrap=True)

    def arrow(p1, p2):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow((1.75, 2.2), (5.25, 3.5))   # X -> M  (path a)
    arrow((6.5, 3.5), (8.75, 2.2))    # M -> Y  (path b)
    arrow((3.0, 1.6), (7.5, 1.6))     # X -> Y  (direct, c')

    ci_excludes_zero = not (result.indirect_ci[0] <= 0 <= result.indirect_ci[1])
    ax.text(3.0, 4.0, f"a = {result.a_coef:+.3g}\n(p={result.a_pvalue:.3g})", fontsize=8, ha="center")
    ax.text(7.0, 4.0, f"b = {result.b_coef:+.3g}\n(p={result.b_pvalue:.3g})", fontsize=8, ha="center")
    ax.text(5.25, 0.9, f"c' (direct) = {result.c_prime:+.3g}", fontsize=8, ha="center")
    ax.text(
        5.0, 5.4,
        f"Indirect (a x b) = {result.indirect_effect:+.3g}  "
        f"95% CI [{result.indirect_ci[0]:+.3g}, {result.indirect_ci[1]:+.3g}]"
        f"  {'(excludes 0)' if ci_excludes_zero else '(includes 0)'}",
        fontsize=9, ha="center", fontweight="bold",
        color="darkgreen" if ci_excludes_zero else "darkred",
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved path diagram to %s", outpath)


def plot_power_curve(power_df: pd.DataFrame, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(power_df["n"], power_df["power"], marker="o")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, label="80% power convention")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Empirical power to detect mediation (indirect-effect CI excludes 0)")
    ax.set_title("Mediation detectability vs. sample size")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved power curve to %s", outpath)


# --------------------------------------------------------------------------- #
# DRIVER
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="Path to real patient-level data (treatment/mediator/outcome columns)")
    parser.add_argument("--treatment", default="treatment")
    parser.add_argument("--mediator", default="mediator")
    parser.add_argument("--outcome", default="outcome")
    parser.add_argument("--covariates", default="", help="Comma-separated covariate column names")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="output/mediation")

    parser.add_argument("--simulate-illustration", action="store_true",
                         help="Run the calibrated-but-illustrative Athauda 2017/2019 scenario "
                              "(NOT real data — see module docstring)")
    parser.add_argument("--assumed-proportion-mediated", type=float, default=0.4)

    parser.add_argument("--power-analysis", action="store_true")
    parser.add_argument("--true-a", type=float, default=0.27)
    parser.add_argument("--true-b", type=float, default=0.5)
    parser.add_argument("--true-cprime", type=float, default=1.75)
    parser.add_argument("--n-list", default="62,150,300,600,1200,3808")
    parser.add_argument("--n-sims", type=int, default=300)

    parser.add_argument("--dry-run", action="store_true",
                         help="Quick offline smoke test on arbitrary simulated data, no calibration claims")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]
        df = pd.read_csv(args.csv)
        result = fit_mediation(df, args.treatment, args.mediator, args.outcome,
                                covariates=covariates, n_boot=args.n_boot, seed=args.seed)
        print(result.summary())
        plot_path_diagram(result, args.treatment, args.mediator, args.outcome,
                           outdir / "path_diagram.png", title=f"Mediation analysis: {args.csv}")

    elif args.simulate_illustration:
        log.warning("=== ILLUSTRATIVE SIMULATION MODE — NOT REAL DATA. See module docstring. ===")
        df, calibration_notes = simulate_athauda2019_illustration(
            assumed_proportion_mediated=args.assumed_proportion_mediated, seed=args.seed
        )
        result = fit_mediation(
            df, "treatment", "mediator_irs1_pTyr_change", "outcome_updrs3_improvement",
            n_boot=args.n_boot, seed=args.seed,
            notes=["THIS IS A CALIBRATED SIMULATION, NOT A REAL FINDING.",
                   f"Path a and total effect are real aggregate statistics; path b was "
                   f"ASSUMED such that {args.assumed_proportion_mediated:.0%} of the total "
                   f"effect is mediated — an arbitrary choice, not a result."],
        )
        print(result.summary())
        print("\nCalibration provenance:")
        for category, values in calibration_notes.items():
            print(f"  {category}:")
            for k, v in values.items():
                print(f"    {k}: {v}")
        df.to_csv(outdir / "athauda2019_illustration_data.csv", index=False)
        plot_path_diagram(result, "Exenatide vs placebo", "IRS-1 pTyr change (exosome)",
                           "MDS-UPDRS III improvement", outdir / "athauda2019_illustration_path_diagram.png",
                           title="ILLUSTRATIVE SIMULATION (not real data) — Athauda 2017/2019-calibrated")

    elif args.power_analysis:
        n_list = [int(x) for x in args.n_list.split(",")]
        power_df = run_power_analysis(
            n_list=n_list, true_a=args.true_a, true_b=args.true_b, true_cprime=args.true_cprime,
            n_sims=args.n_sims, seed=args.seed,
        )
        print(power_df.to_string(index=False))
        power_df.to_csv(outdir / "power_analysis.csv", index=False)
        plot_power_curve(power_df, outdir / "power_curve.png")

    elif args.dry_run:
        log.info("=== DRY RUN: arbitrary synthetic data, mechanism smoke-test only ===")
        df = simulate_mediation_dataset(n=300, true_a=1.0, true_b=0.8, true_cprime=0.3, seed=args.seed)
        result = fit_mediation(df, "treatment", "mediator", "outcome", n_boot=2000, seed=args.seed)
        print(result.summary())
        plot_path_diagram(result, "treatment", "mediator", "outcome", outdir / "dry_run_path_diagram.png")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
