"""
Offline pytest suite for mediation_model.py. Everything here uses
simulate_mediation_dataset() with known ground-truth effects, so we can check
the model RECOVERS known truth — this validates the statistical machinery
itself, not any claim about GLP-1RAs or insulin resistance.

Run with: pytest analysis/tests/test_mediation_model.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import mediation_model as mm


# --------------------------------------------------------------------------- #
# Core mediation fitting: does it recover known ground truth?
# --------------------------------------------------------------------------- #

def test_fit_mediation_recovers_strong_known_mediation():
    df = mm.simulate_mediation_dataset(
        n=2000, true_a=2.0, true_b=1.5, true_cprime=0.5,
        mediator_noise_sd=1.0, outcome_noise_sd=1.0, seed=1,
    )
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=500, seed=1)

    assert result.n == 2000
    assert abs(result.a_coef - 2.0) < 0.2
    assert abs(result.b_coef - 1.5) < 0.2
    assert abs(result.c_prime - 0.5) < 0.3
    expected_indirect = 2.0 * 1.5
    assert abs(result.indirect_effect - expected_indirect) < 0.5
    # With a strong true effect and n=2000, the bootstrap CI should clearly exclude 0
    assert result.indirect_ci[0] > 0


def test_fit_mediation_null_mediation_ci_includes_zero():
    # true_a = 0: treatment has no effect on the mediator at all -> no mediation possible
    df = mm.simulate_mediation_dataset(
        n=500, true_a=0.0, true_b=1.5, true_cprime=1.0,
        mediator_noise_sd=1.0, outcome_noise_sd=1.0, seed=2,
    )
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=500, seed=2)
    assert result.indirect_ci[0] <= 0 <= result.indirect_ci[1]


def test_total_effect_equals_direct_plus_indirect():
    """Baron-Kenny algebraic identity: c (total) == c' (direct) + a*b (indirect),
    exactly, for OLS on the same sample."""
    df = mm.simulate_mediation_dataset(n=400, true_a=1.2, true_b=0.7, true_cprime=0.4, seed=3)
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=100, seed=3)
    assert abs(result.total_effect - (result.c_prime + result.indirect_effect)) < 1e-8


def test_proportion_mediated_is_fraction_of_total():
    df = mm.simulate_mediation_dataset(n=1000, true_a=1.0, true_b=1.0, true_cprime=1.0, seed=4)
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=200, seed=4)
    assert 0.0 < result.proportion_mediated < 1.0  # roughly half mediated, half direct by construction


def test_fit_mediation_raises_on_too_few_rows():
    df = mm.simulate_mediation_dataset(n=5, seed=5)
    with pytest.raises(ValueError):
        mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=50, seed=5)


def test_fit_mediation_with_covariates_runs():
    df = mm.simulate_mediation_dataset(n=300, true_a=1.0, true_b=1.0, true_cprime=0.5, seed=6)
    df["age"] = np.random.default_rng(6).normal(60, 10, size=len(df))
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome",
                               covariates=["age"], n_boot=200, seed=6)
    assert result.n == 300


# --------------------------------------------------------------------------- #
# Calibrated illustration — check it's deterministic and clearly self-flagged
# --------------------------------------------------------------------------- #

def test_simulate_athauda2019_illustration_uses_real_n():
    df, notes = mm.simulate_athauda2019_illustration(seed=7)
    assert len(df) == 62
    assert df["treatment"].sum() == 32  # 32 exenatide
    assert (df["treatment"] == 0).sum() == 30  # 30 placebo


def test_simulate_athauda2019_illustration_flags_assumptions():
    _, notes = mm.simulate_athauda2019_illustration(seed=7)
    assert "real_calibrated_values" in notes
    assert "assumed_illustrative_values" in notes
    assert "warning" in notes["assumed_illustrative_values"]
    # the real path-a target must match the literature number exactly
    assert notes["real_calibrated_values"]["path_a_target_AU"] == 0.27


def test_simulate_athauda2019_illustration_sensitive_to_assumption():
    """The whole point of flagging path b as assumed: changing the assumption
    should visibly change the 'result', proving it's not derived from data."""
    df_low, _ = mm.simulate_athauda2019_illustration(assumed_proportion_mediated=0.1, seed=8)
    df_high, _ = mm.simulate_athauda2019_illustration(assumed_proportion_mediated=0.9, seed=8)
    result_low = mm.fit_mediation(df_low, "treatment", "mediator_irs1_pTyr_change",
                                   "outcome_updrs3_improvement", n_boot=200, seed=8)
    result_high = mm.fit_mediation(df_high, "treatment", "mediator_irs1_pTyr_change",
                                    "outcome_updrs3_improvement", n_boot=200, seed=8)
    assert result_low.proportion_mediated < result_high.proportion_mediated


def test_simulate_athauda2019_illustration_runs_through_fit_mediation():
    df, _ = mm.simulate_athauda2019_illustration(seed=9)
    result = mm.fit_mediation(df, "treatment", "mediator_irs1_pTyr_change",
                               "outcome_updrs3_improvement", n_boot=300, seed=9)
    assert result.n == 62


# --------------------------------------------------------------------------- #
# Power analysis
# --------------------------------------------------------------------------- #

def test_power_analysis_increases_with_sample_size():
    power_df = mm.run_power_analysis(
        n_list=[20, 500], true_a=0.5, true_b=0.5, true_cprime=0.2,
        n_sims=40, n_boot=300, seed=10,
    )
    power_small = power_df.loc[power_df["n"] == 20, "power"].iloc[0]
    power_large = power_df.loc[power_df["n"] == 500, "power"].iloc[0]
    assert power_large >= power_small


def test_power_analysis_near_zero_for_no_true_effect():
    power_df = mm.run_power_analysis(
        n_list=[100], true_a=0.0, true_b=0.0, true_cprime=1.0,
        n_sims=40, n_boot=300, seed=11,
    )
    # with no true mediation at all, false-positive rate should be roughly the
    # nominal ~5% level, generously bounded here to avoid test flakiness
    assert power_df["power"].iloc[0] < 0.3


# --------------------------------------------------------------------------- #
# Plotting (smoke tests)
# --------------------------------------------------------------------------- #

def test_plot_path_diagram_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    df = mm.simulate_mediation_dataset(n=200, true_a=1.0, true_b=1.0, true_cprime=0.5, seed=12)
    result = mm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=200, seed=12)
    outpath = tmp_path / "path.png"
    mm.plot_path_diagram(result, "X", "M", "Y", outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_power_curve_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    power_df = mm.run_power_analysis(n_list=[30, 100], true_a=0.5, true_b=0.5, true_cprime=0.2,
                                      n_sims=20, n_boot=200, seed=13)
    outpath = tmp_path / "power.png"
    mm.plot_power_curve(power_df, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0
