"""
Offline pytest suite for geo_de_pathway_pipeline.py.

Everything here runs against generate_mock_dataset() / mock=True code paths only —
no network access (no GEO, no Enrichr, no KEGG REST calls). Run with:

    pip install -r requirements.txt
    pytest analysis/tests/ -v

These tests check the *pipeline logic* (DE math, enrichment math, gene matching,
plotting), not any specific biological claim about a real GEO dataset.
"""
from __future__ import annotations

import pandas as pd
import pytest

import geo_de_pathway_pipeline as pl


# --------------------------------------------------------------------------- #
# 1. Mock data generator
# --------------------------------------------------------------------------- #

def test_generate_mock_dataset_shapes_and_groups():
    expr, meta, truth = pl.generate_mock_dataset(
        species="mouse", data_kind="counts", n_decoy_genes=100,
        n_samples_per_group=4, seed=1,
    )
    n_genes_expected = len(truth["insulin_genes_in_universe"]) + len(truth["decoy_genes_in_universe"])
    assert expr.shape == (n_genes_expected, 8)
    assert list(meta["group"]).count("control") == 4
    assert list(meta["group"]).count("treatment") == 4
    assert set(expr.columns) == set(meta.index)
    # counts must be non-negative integers (Poisson-generated)
    assert (expr.values >= 0).all()
    assert (expr.values == expr.values.round()).all()


def test_generate_mock_dataset_is_deterministic_given_seed():
    expr1, _, _ = pl.generate_mock_dataset(seed=123)
    expr2, _, _ = pl.generate_mock_dataset(seed=123)
    pd.testing.assert_frame_equal(expr1, expr2)


def test_generate_mock_dataset_normalized_kind_is_continuous():
    expr, _, _ = pl.generate_mock_dataset(data_kind="normalized", seed=2)
    # normalized/log-intensity data should NOT all be integers (vs. the counts path)
    assert not (expr.values == expr.values.round()).all()


# --------------------------------------------------------------------------- #
# 2. KEGG fetch (mock mode = no network)
# --------------------------------------------------------------------------- #

def test_fetch_kegg_pathway_genes_mock_makes_no_network_call(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("requests.get should not be called when mock=True")

    monkeypatch.setattr(pl.requests, "get", _fail_if_called)
    genes = pl.fetch_kegg_pathway_genes("hsa04910", mock=True)
    assert genes == pl.FALLBACK_INSULIN_SIGNALING_GENES_HUMAN


def test_fetch_kegg_pathway_genes_mock_mouse_is_titlecased(monkeypatch):
    monkeypatch.setattr(pl.requests, "get",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    genes = pl.fetch_kegg_pathway_genes("mmu04910", mock=True)
    assert "Insr" in genes
    assert "INSR" not in genes  # should be mouse-cased, not human all-caps


# --------------------------------------------------------------------------- #
# 3. Differential expression (both code paths)
# --------------------------------------------------------------------------- #

def test_de_normalized_path_recovers_known_perturbed_genes():
    """Welch's t-test + FDR fallback path (no heavy optional dependency needed)."""
    expr, meta, truth = pl.generate_mock_dataset(
        data_kind="normalized", n_decoy_genes=200, n_samples_per_group=8,
        n_true_de_insulin_genes=8, effect_size=2.0, seed=10,
    )
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")

    assert {"log2FoldChange", "pvalue", "padj"}.issubset(de.columns)
    assert len(de) == len(expr)

    true_de = set(truth["true_de_insulin_genes"]) | set(truth["true_de_decoy_genes"])
    detected = set(de.index[de["padj"] < 0.05])
    recall = len(detected & true_de) / len(true_de)
    assert recall > 0.5, f"Expected most deliberately-perturbed genes to be detected, got recall={recall}"


def test_de_counts_path_with_pydeseq2():
    pytest.importorskip("pydeseq2")
    expr, meta, truth = pl.generate_mock_dataset(
        data_kind="counts", n_decoy_genes=150, n_samples_per_group=6,
        n_true_de_insulin_genes=6, effect_size=2.0, seed=11,
    )
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "counts")
    assert {"log2FoldChange", "padj"}.issubset(de.columns)

    true_de = set(truth["true_de_insulin_genes"]) | set(truth["true_de_decoy_genes"])
    detected = set(de.index[de["padj"] < 0.05])
    recall = len(detected & true_de) / len(true_de)
    assert recall > 0.3, f"Expected PyDESeq2 to recover a good fraction of true DE genes, got {recall}"


def test_de_raises_on_more_than_two_groups():
    expr, meta, _ = pl.generate_mock_dataset(seed=3)
    meta = meta.copy()
    meta.iloc[0, 0] = "third_group"
    with pytest.raises(ValueError):
        pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")


# --------------------------------------------------------------------------- #
# 4. Insulin-signaling-gene highlighting
# --------------------------------------------------------------------------- #

def test_highlight_insulin_signaling_genes_matches_expected_set():
    expr, meta, truth = pl.generate_mock_dataset(data_kind="normalized", seed=4)
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    subset = pl.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    # Independently re-derive the expected gene set the same way highlight_insulin_signaling_genes
    # does internally, to check it actually used/matched against that set (not a tautology, since
    # we're asserting equality with de.index restricted to genes generate_mock_dataset put in).
    expected_insulin_genes = set(pl.fetch_kegg_pathway_genes("mmu04910", mock=True))
    assert set(subset.index) == expected_insulin_genes & set(de.index)
    assert "in_insulin_signaling_pathway" in de.columns
    assert de.loc[list(subset.index), "in_insulin_signaling_pathway"].all()


def test_highlight_insulin_signaling_genes_recovers_perturbed_ones():
    expr, meta, truth = pl.generate_mock_dataset(
        data_kind="normalized", n_true_de_insulin_genes=10, effect_size=2.5, seed=5
    )
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    subset = pl.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    # generate_mock_dataset(species="mouse") already returns mouse-cased gene symbols
    # (via fetch_kegg_pathway_genes("mmu04910", mock=True)), matching subset.index directly.
    true_de_insulin = set(truth["true_de_insulin_genes"])
    detected_sig = set(subset.index[subset["padj"] < 0.05])
    assert len(detected_sig & true_de_insulin) > 0, (
        "Expected at least some deliberately-perturbed insulin genes to surface as DE"
    )


# --------------------------------------------------------------------------- #
# 5. Offline pathway enrichment
# --------------------------------------------------------------------------- #

def test_offline_enrichment_flags_insulin_pathway_as_significant():
    # Force a strong, clean insulin-pathway signal: perturb every insulin-pathway gene.
    n_insulin_genes = len(pl.fetch_kegg_pathway_genes("mmu04910", mock=True))
    expr, meta, truth = pl.generate_mock_dataset(
        data_kind="normalized", n_decoy_genes=200,
        n_true_de_insulin_genes=n_insulin_genes,
        n_true_de_decoy_genes=2, effect_size=3.0, seed=6,
    )
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    enrichment = pl.run_pathway_enrichment_offline(de, species="mouse")

    assert not enrichment.empty
    assert {"Term", "Adjusted P-value", "is_insulin_related_pathway"}.issubset(enrichment.columns)

    insulin_row = enrichment[enrichment["Term"].str.contains("Insulin", case=False)]
    assert not insulin_row.empty
    assert insulin_row.iloc[0]["is_insulin_related_pathway"]
    assert insulin_row.iloc[0]["Adjusted P-value"] < 0.05, (
        "Insulin signaling pathway should be flagged as significantly enriched "
        "when nearly all its genes are deliberately perturbed"
    )


def test_offline_enrichment_empty_when_no_overlap():
    expr, meta, _ = pl.generate_mock_dataset(data_kind="normalized", seed=8)
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    # Gene sets that share nothing with the dataset's gene universe
    empty_sets = {"Nonexistent pathway": ["NotAGene1", "NotAGene2"]}
    enrichment = pl.run_pathway_enrichment_offline(de, species="mouse", gene_sets=empty_sets)
    assert enrichment.empty


# --------------------------------------------------------------------------- #
# 6. Plotting (smoke tests — just confirm files are produced without error)
# --------------------------------------------------------------------------- #

def test_plot_volcano_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    expr, meta, _ = pl.generate_mock_dataset(data_kind="normalized", seed=9)
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    pl.highlight_insulin_signaling_genes(de, species="mouse", mock=True)

    outpath = tmp_path / "volcano.png"
    pl.plot_volcano(de, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_enrichment_barplot_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    expr, meta, _ = pl.generate_mock_dataset(data_kind="normalized", seed=12)
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    enrichment = pl.run_pathway_enrichment_offline(de, species="mouse")
    if enrichment.empty:
        pytest.skip("No enrichment results generated for this seed; nothing to plot")

    outpath = tmp_path / "enrichment.png"
    pl.plot_enrichment_barplot(enrichment, outpath)
    assert outpath.exists() and outpath.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 7. Full offline end-to-end smoke test (mirrors `--dry-run` CLI flow)
# --------------------------------------------------------------------------- #

def test_full_dry_run_pipeline_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    expr, meta, truth = pl.generate_mock_dataset(
        data_kind="normalized", n_true_de_insulin_genes=6, effect_size=2.0, seed=42
    )
    de = pl.run_differential_expression(expr, meta, "group", "control", "treatment", "normalized")
    insulin_subset = pl.highlight_insulin_signaling_genes(de, species="mouse", mock=True)
    enrichment = pl.run_pathway_enrichment_offline(de, species="mouse")

    pl.plot_volcano(de, tmp_path / "volcano.png")
    if not enrichment.empty:
        pl.plot_enrichment_barplot(enrichment, tmp_path / "enrichment.png")

    assert len(insulin_subset) > 0
    assert (tmp_path / "volcano.png").exists()
