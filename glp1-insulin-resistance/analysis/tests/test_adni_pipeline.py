"""
Offline pytest suite for adni_pipeline.py. Everything here uses
generate_synthetic_adni_demo() with a known injected mediation structure, so
we can check the load -> merge -> mediate pipeline RECOVERS known truth — this
validates the plumbing, not any claim about real ADNI subjects.

Run with: pytest analysis/tests/test_adni_pipeline.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

import adni_pipeline as ap


def test_compute_homa_ir_matches_formula():
    glucose = pd.Series([100.0])
    insulin = pd.Series([10.0])
    result = ap.compute_homa_ir(glucose, insulin)
    assert abs(result.iloc[0] - (100.0 * 10.0 / 405.0)) < 1e-9


def test_flag_glp1ra_users_matches_known_drug_names():
    meds = pd.DataFrame({
        "RID": [1, 1, 2, 3],
        "CMMED": ["Liraglutide 1.8mg", "Aspirin", "Metformin", "Ozempic"],
    })
    flags = ap.flag_glp1ra_users(meds)
    flagged = dict(zip(flags["RID"], flags["glp1ra_user"]))
    assert flagged[1] is True   # liraglutide
    assert flagged[2] is False  # metformin only, not a GLP-1RA
    assert flagged[3] is True   # ozempic (semaglutide brand)


def test_flag_glp1ra_users_no_false_positive_on_unrelated_meds():
    meds = pd.DataFrame({"RID": [1, 2], "CMMED": ["Donepezil", "Memantine"]})
    flags = ap.flag_glp1ra_users(meds)
    assert not flags["glp1ra_user"].any()


def test_generate_synthetic_adni_demo_produces_expected_tables():
    adnimerge, medications, biomarkers = ap.generate_synthetic_adni_demo(n=100, seed=1)
    assert adnimerge["RID"].nunique() == 100
    assert set(adnimerge["RID"]) == set(medications["RID"]) == set(biomarkers["RID"])
    assert {"GLUCOSE", "INSULIN"}.issubset(biomarkers.columns)


def test_build_cohort_recovers_injected_mediation_direction():
    adnimerge, medications, biomarkers = ap.generate_synthetic_adni_demo(
        n=500, true_a=-1.0, true_b=0.5, true_cprime=0.0, seed=2,
    )
    glp1ra_flags = ap.flag_glp1ra_users(medications)
    homa_ir = biomarkers.assign(
        homa_ir=ap.compute_homa_ir(biomarkers["GLUCOSE"], biomarkers["INSULIN"])
    )[["RID", "homa_ir"]]
    cohort = ap.build_cohort(adnimerge, glp1ra_flags, homa_ir, outcome="adas13")

    assert len(cohort) == 500
    assert {"treatment", "mediator", "outcome_change"}.issubset(cohort.columns)
    assert cohort["treatment"].isin([0.0, 1.0]).all()

    import mediation_model as mm
    result = mm.fit_mediation(cohort, "treatment", "mediator", "outcome_change", n_boot=500, seed=2)
    # true_a=-1.0 (treatment lowers HOMA-IR), true_b=+0.5 (higher HOMA-IR worsens outcome)
    # => indirect effect should be negative (treatment -> lower HOMA-IR -> better outcome)
    assert result.indirect_effect < 0


def test_build_cohort_drops_single_visit_subjects(caplog):
    adnimerge, medications, biomarkers = ap.generate_synthetic_adni_demo(n=50, seed=3)
    single_visit = adnimerge.groupby("RID").head(1)
    glp1ra_flags = ap.flag_glp1ra_users(medications)
    homa_ir = biomarkers.assign(
        homa_ir=ap.compute_homa_ir(biomarkers["GLUCOSE"], biomarkers["INSULIN"])
    )[["RID", "homa_ir"]]
    cohort = ap.build_cohort(single_visit, glp1ra_flags, homa_ir, outcome="adas13")
    assert len(cohort) == 0


def test_load_biomarkers_raises_on_missing_column(tmp_path):
    csv_path = tmp_path / "biomarkers.csv"
    pd.DataFrame({"RID": [1, 2], "GLUCOSE": [100, 110]}).to_csv(csv_path, index=False)
    with pytest.raises(KeyError):
        ap.load_biomarkers(csv_path)


def test_plot_cohort_overview_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    adnimerge, medications, biomarkers = ap.generate_synthetic_adni_demo(n=100, seed=4)
    glp1ra_flags = ap.flag_glp1ra_users(medications)
    homa_ir = biomarkers.assign(
        homa_ir=ap.compute_homa_ir(biomarkers["GLUCOSE"], biomarkers["INSULIN"])
    )[["RID", "homa_ir"]]
    cohort = ap.build_cohort(adnimerge, glp1ra_flags, homa_ir, outcome="adas13")
    outpath = tmp_path / "overview.png"
    ap.plot_cohort_overview(cohort, "Change in ADAS13", outpath)
    assert outpath.exists() and outpath.stat().st_size > 0
