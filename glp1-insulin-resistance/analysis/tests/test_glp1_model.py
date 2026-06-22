"""
Offline pytest suite for glp1_model.py (the merged GEO + mediation + integrated
pipeline model). Everything here runs with zero network calls — no GEO, no
Enrichr, no KEGG REST, no real patient data. Run with:

    pip install -r requirements.txt
    pytest analysis/tests/test_glp1_model.py -v

Organized in three sections mirroring the module's three stages:
  A. GEO transcriptomics pipeline (DE, enrichment, insulin-gene highlighting)
  B. Causal mediation model (path fitting, simulation, power analysis)
  C. Integrated pipeline demo (Stage A output feeding Stage B input)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import glp1_model as gm


# =========================================================================== #
# STAGE A — GEO PIPELINE TESTS
# =========================================================================== #

# --------------------------------------------------------------------------- #
# A1. Mock data generator
# --------------------------------------------------------------------------- #

def test_generate_mock_dataset_shapes_and_groups():
    expr, meta, truth = gm.generate_mock_dataset(
        species="mouse", data_kind="counts", n_decoy_genes=100,
        n_samples_per_group=4, seed=1,
    )
    n_genes_expected = len(truth["insulin_genes_in_universe"]) + len(truth["decoy_genes_in_universe"])
    assert expr.shape == (n_genes_expected, 8)
    assert list(meta["group"]).count("control") == 4
    assert list(meta["group"]).count("treatment") == 4
    assert set(expr.columns) == set(meta.index)
    assert (expr.values >= 0).all()
    assert (expr.values == expr.values.round()).all()


def test_generate_mock_dataset_is_deterministic_given_seed():
    expr1, _, _ = gm.generate_mock_dataset(seed=123)
    expr2, _, _ = gm.generate_mock_dataset(seed=123)
    pd.testing.assert_frame_equal(expr1, expr2)


def test_generate_mock_dataset_normalized_kind_is_continuous():
    expr, _, _ = gm.generate_mock_dataset(data_kind="normalized", seed=2)
    assert not (expr.values == expr.values.round()).all()


# --------------------------------------------------------------------------- #
# A2. KEGG fetch (mock mode = no network)
# --------------------------------------------------------------------------- #

def test_fetch_kegg_pathway_genes_mock_makes_no_network_call(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("requests.get should not be called when mock=True")

    monkeypatch.setattr(gm.requests, "get", _fail_if_called)
    genes = gm.fetch_kegg_pathway_genes("hsa04910", mock=True)
    assert genes == gm.FALLBACK_INSULIN_SIGNALING_GENES_HUMAN


def test_fetch_kegg_pathway_genes_mock_mouse_is_titlecased(monkeypatch):
    monkeypatch.setattr(gm.requests, "get",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    genes = gm.fetch_kegg_pathway_genes("mmu04910", mock=True)
    assert "Insr" in genes
    assert "INSR" not in genes


# --------------------------------------------------------------------------- #
# A3. Differential expression (both code paths)
# --------------------------------------------------------------------------- #

def test_de_normalized_path_recovers_known_perturbed_genes():
    expr, meta, truth = gm.generate_mock_dataset(
        data_kind="normalized", n_decoy_genes=200, n_samples_per_group=8,
        n_true_de_insulin_genes=8, effect_size=2.0, seed=10,
    )
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")

    assert {"log2FoldChange", "pvalue", "padj"}.issubset(de.columns)
    assert len(de) == len(expr)

    true_de = set(truth["true_de_insulin_genes"]) | set(truth["true_de_decoy_genes"])
    detected = set(de.index[de["padj"] < 0.05])
    recall = len(detected & true_de) / len(true_de)
    assert recall > 0.5


def test_de_counts_path_with_pydeseq2():
    pytest.importorskip("pydeseq2")
    expr, meta, truth = gm.generate_mock_dataset(
        data_kind="counts", n_decoy_genes=150, n_samples_per_group=6,
        n_true_de_insulin_genes=6, effect_size=2.0, seed=11,
    )
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "counts")
    assert {"log2FoldChange", "padj"}.issubset(de.columns)

    true_de = set(truth["true_de_insulin_genes"]) | set(truth["true_de_decoy_genes"])
    detected = set(de.index[de["padj"] < 0.05])
    recall = len(detected & true_de) / len(true_de)
    assert recall > 0.3


def test_de_raises_when_only_one_group_present_after_filtering():
    """run_differential_expression() filters samples to isin([control, treatment])
    BEFORE checking for exactly 2 groups, so an unrelated third label elsewhere in
    the metadata gets filtered out and never reaches the check. The check instead
    guards against the filtered set collapsing to 0 or 1 distinct label — e.g. if
    one arm has no samples at all, as constructed here."""
    expr, meta, _ = gm.generate_mock_dataset(seed=3)
    meta = meta.copy()
    meta["group"] = "control"  # no "treatment" samples exist anymore
    with pytest.raises(ValueError):
        gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")


# --------------------------------------------------------------------------- #
# A4. Insulin-signaling-gene highlighting
# --------------------------------------------------------------------------- #

def test_highlight_insulin_signaling_genes_matches_expected_set():
    expr, meta, truth = gm.generate_mock_dataset(data_kind="normalized", seed=4)
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    subset = gm.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    expected_insulin_genes = set(gm.fetch_kegg_pathway_genes("mmu04910", mock=True))
    assert set(subset.index) == expected_insulin_genes & set(de.index)
    assert "in_insulin_signaling_pathway" in de.columns
    assert de.loc[list(subset.index), "in_insulin_signaling_pathway"].all()


def test_highlight_insulin_signaling_genes_recovers_perturbed_ones():
    expr, meta, truth = gm.generate_mock_dataset(
        data_kind="normalized", n_true_de_insulin_genes=10, effect_size=2.5, seed=5
    )
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    subset = gm.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    true_de_insulin = set(truth["true_de_insulin_genes"])
    detected_sig = set(subset.index[subset["padj"] < 0.05])
    assert len(detected_sig & true_de_insulin) > 0


# --------------------------------------------------------------------------- #
# A5. Offline pathway enrichment
# --------------------------------------------------------------------------- #

def test_offline_enrichment_flags_insulin_pathway_as_significant():
    n_insulin_genes = len(gm.fetch_kegg_pathway_genes("mmu04910", mock=True))
    expr, meta, truth = gm.generate_mock_dataset(
        data_kind="normalized", n_decoy_genes=200,
        n_true_de_insulin_genes=n_insulin_genes,
        n_true_de_decoy_genes=2, effect_size=3.0, seed=6,
    )
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    enrichment = gm.run_pathway_enrichment_offline(de, species="mouse")

    assert not enrichment.empty
    assert {"Term", "Adjusted P-value", "is_insulin_related_pathway"}.issubset(enrichment.columns)

    insulin_row = enrichment[enrichment["Term"].str.contains("Insulin", case=False)]
    assert not insulin_row.empty
    assert insulin_row.iloc[0]["is_insulin_related_pathway"]
    assert insulin_row.iloc[0]["Adjusted P-value"] < 0.05


def test_offline_enrichment_empty_when_no_overlap():
    expr, meta, _ = gm.generate_mock_dataset(data_kind="normalized", seed=8)
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    empty_sets = {"Nonexistent pathway": ["NotAGene1", "NotAGene2"]}
    enrichment = gm.run_pathway_enrichment_offline(de, species="mouse", gene_sets=empty_sets)
    assert enrichment.empty


# --------------------------------------------------------------------------- #
# A6. Plotting (smoke tests)
# --------------------------------------------------------------------------- #

def test_plot_volcano_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    expr, meta, _ = gm.generate_mock_dataset(data_kind="normalized", seed=9)
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    gm.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    outpath = tmp_path / "volcano.png"
    gm.plot_volcano(de, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_enrichment_barplot_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    expr, meta, _ = gm.generate_mock_dataset(data_kind="normalized", seed=12)
    de = gm.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    enrichment = gm.run_pathway_enrichment_offline(de, species="mouse")
    if enrichment.empty:
        pytest.skip("No enrichment results generated for this seed; nothing to plot")

    outpath = tmp_path / "enrichment.png"
    gm.plot_enrichment_barplot(enrichment, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


# --------------------------------------------------------------------------- #
# A7. run_geo() end-to-end (dry-run CLI flow)
# --------------------------------------------------------------------------- #

def test_run_geo_dry_run_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    args = gm.argparse.Namespace(
        dry_run=True, dry_run_data_kind="normalized", dry_run_seed=42,
        species="mouse", outdir="output", padj=0.05, log2fc=0.5,
    )
    result = gm.run_geo(args)
    assert "de_results" in result and "insulin_subset" in result
    assert (tmp_path / "output" / "MOCK" / "de_results.csv").exists()


# =========================================================================== #
# STAGE B — MEDIATION MODEL TESTS
# =========================================================================== #

# --------------------------------------------------------------------------- #
# B1. Core mediation fitting: does it recover known ground truth?
# --------------------------------------------------------------------------- #

def test_fit_mediation_recovers_strong_known_mediation():
    df = gm.simulate_mediation_dataset(
        n=2000, true_a=2.0, true_b=1.5, true_cprime=0.5,
        mediator_noise_sd=1.0, outcome_noise_sd=1.0, seed=1,
    )
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=500, seed=1)

    assert result.n == 2000
    assert abs(result.a_coef - 2.0) < 0.2
    assert abs(result.b_coef - 1.5) < 0.2
    assert abs(result.c_prime - 0.5) < 0.3
    expected_indirect = 2.0 * 1.5
    assert abs(result.indirect_effect - expected_indirect) < 0.5
    assert result.indirect_ci[0] > 0


def test_fit_mediation_null_mediation_ci_includes_zero():
    df = gm.simulate_mediation_dataset(
        n=500, true_a=0.0, true_b=1.5, true_cprime=1.0,
        mediator_noise_sd=1.0, outcome_noise_sd=1.0, seed=2,
    )
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=500, seed=2)
    assert result.indirect_ci[0] <= 0 <= result.indirect_ci[1]


def test_total_effect_equals_direct_plus_indirect():
    df = gm.simulate_mediation_dataset(n=400, true_a=1.2, true_b=0.7, true_cprime=0.4, seed=3)
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=100, seed=3)
    assert abs(result.total_effect - (result.c_prime + result.indirect_effect)) < 1e-8


def test_proportion_mediated_is_fraction_of_total():
    df = gm.simulate_mediation_dataset(n=1000, true_a=1.0, true_b=1.0, true_cprime=1.0, seed=4)
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=200, seed=4)
    assert 0.0 < result.proportion_mediated < 1.0


def test_fit_mediation_raises_on_too_few_rows():
    df = gm.simulate_mediation_dataset(n=5, seed=5)
    with pytest.raises(ValueError):
        gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=50, seed=5)


def test_fit_mediation_with_covariates_runs():
    df = gm.simulate_mediation_dataset(n=300, true_a=1.0, true_b=1.0, true_cprime=0.5, seed=6)
    df["age"] = np.random.default_rng(6).normal(60, 10, size=len(df))
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome",
                               covariates=["age"], n_boot=200, seed=6)
    assert result.n == 300


# --------------------------------------------------------------------------- #
# B2. Calibrated illustration — check it's deterministic and clearly self-flagged
# --------------------------------------------------------------------------- #

def test_simulate_athauda2019_illustration_uses_real_n():
    df, notes = gm.simulate_athauda2019_illustration(seed=7)
    assert len(df) == 62
    assert df["treatment"].sum() == 32
    assert (df["treatment"] == 0).sum() == 30


def test_simulate_athauda2019_illustration_flags_assumptions():
    _, notes = gm.simulate_athauda2019_illustration(seed=7)
    assert "real_calibrated_values" in notes
    assert "assumed_illustrative_values" in notes
    assert "warning" in notes["assumed_illustrative_values"]
    assert notes["real_calibrated_values"]["path_a_target_AU"] == 0.27


def test_simulate_athauda2019_illustration_sensitive_to_assumption():
    df_low, _ = gm.simulate_athauda2019_illustration(assumed_proportion_mediated=0.1, seed=8)
    df_high, _ = gm.simulate_athauda2019_illustration(assumed_proportion_mediated=0.9, seed=8)
    result_low = gm.fit_mediation(df_low, "treatment", "mediator_irs1_pTyr_change",
                                   "outcome_updrs3_improvement", n_boot=200, seed=8)
    result_high = gm.fit_mediation(df_high, "treatment", "mediator_irs1_pTyr_change",
                                    "outcome_updrs3_improvement", n_boot=200, seed=8)
    assert result_low.proportion_mediated < result_high.proportion_mediated


def test_simulate_athauda2019_illustration_runs_through_fit_mediation():
    df, _ = gm.simulate_athauda2019_illustration(seed=9)
    result = gm.fit_mediation(df, "treatment", "mediator_irs1_pTyr_change",
                               "outcome_updrs3_improvement", n_boot=300, seed=9)
    assert result.n == 62


# --------------------------------------------------------------------------- #
# B3. Power analysis
# --------------------------------------------------------------------------- #

def test_power_analysis_increases_with_sample_size():
    power_df = gm.run_power_analysis(
        n_list=[20, 500], true_a=0.5, true_b=0.5, true_cprime=0.2,
        n_sims=40, n_boot=300, seed=10,
    )
    power_small = power_df.loc[power_df["n"] == 20, "power"].iloc[0]
    power_large = power_df.loc[power_df["n"] == 500, "power"].iloc[0]
    assert power_large >= power_small


def test_power_analysis_near_zero_for_no_true_effect():
    power_df = gm.run_power_analysis(
        n_list=[100], true_a=0.0, true_b=0.0, true_cprime=1.0,
        n_sims=40, n_boot=300, seed=11,
    )
    assert power_df["power"].iloc[0] < 0.3


# --------------------------------------------------------------------------- #
# B4. Plotting (smoke tests)
# --------------------------------------------------------------------------- #

def test_plot_path_diagram_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    df = gm.simulate_mediation_dataset(n=200, true_a=1.0, true_b=1.0, true_cprime=0.5, seed=12)
    result = gm.fit_mediation(df, "treatment", "mediator", "outcome", n_boot=200, seed=12)
    outpath = tmp_path / "path.png"
    gm.plot_path_diagram(result, "X", "M", "Y", outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_power_curve_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    power_df = gm.run_power_analysis(n_list=[30, 100], true_a=0.5, true_b=0.5, true_cprime=0.2,
                                      n_sims=20, n_boot=200, seed=13)
    outpath = tmp_path / "power.png"
    gm.plot_power_curve(power_df, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


# --------------------------------------------------------------------------- #
# B5. run_mediation() end-to-end (dry-run CLI flow)
# --------------------------------------------------------------------------- #

def test_run_mediation_dry_run_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    args = gm.argparse.Namespace(
        csv=None, treatment="treatment", mediator="mediator", outcome="outcome",
        covariates="", n_boot=200, seed=42, outdir="output/mediation",
        simulate_illustration=False, assumed_proportion_mediated=0.4,
        power_analysis=False, true_a=0.27, true_b=0.5, true_cprime=1.75,
        n_list="62,150", n_sims=10, dry_run=True,
    )
    result = gm.run_mediation(args)
    assert result is not None
    assert (tmp_path / "output" / "mediation" / "dry_run_path_diagram.png").exists()


# =========================================================================== #
# STAGE C — INTEGRATED PIPELINE DEMO TESTS
# =========================================================================== #

def test_run_pipeline_demo_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    args = gm.argparse.Namespace(
        species="mouse", seed=42, outdir="output/pipeline",
        mediation_n=150, assumed_true_b=0.8, assumed_true_cprime=0.3, dry_run=True,
    )
    outputs = gm.run_pipeline_demo(args)

    assert "geo_outputs" in outputs and "mediation_result" in outputs
    assert outputs["mediation_result"].n == 150
    assert outputs["handoff_effect_size"] >= 0
    assert (tmp_path / "output" / "pipeline" / "pipeline_demo_path_diagram.png").exists()


def test_run_pipeline_demo_handoff_uses_geo_effect_size(tmp_path, monkeypatch):
    """The path-a magnitude fed into Stage B should come from Stage A's mock
    DE results, not be hardcoded — check it varies with the GEO seed."""
    monkeypatch.chdir(tmp_path)
    args1 = gm.argparse.Namespace(species="mouse", seed=1, outdir="output_a",
                                   mediation_n=100, assumed_true_b=0.5, assumed_true_cprime=0.2,
                                   dry_run=True)
    args2 = gm.argparse.Namespace(species="mouse", seed=99, outdir="output_b",
                                   mediation_n=100, assumed_true_b=0.5, assumed_true_cprime=0.2,
                                   dry_run=True)
    out1 = gm.run_pipeline_demo(args1)
    out2 = gm.run_pipeline_demo(args2)
    # Different seeds -> different mock GEO data -> (very likely) different handoff gene/effect size
    assert (out1["handoff_gene"], out1["handoff_effect_size"]) != (out2["handoff_gene"], out2["handoff_effect_size"])
