"""
GEO expression analysis pipeline: download -> differential expression -> pathway
enrichment -> insulin-signaling-gene highlighting.

Context (see ../extraction/literature_extraction.json -> additional_database_sources.GEO_transcriptomics):
No GEO dataset currently combines AD + insulin signaling + GLP-1 pathway treatment
in one experiment. This pipeline is written generically so it can be pointed at:
  - GSE306976 (default): J20 AD mouse model, pre- vs post-amyloid-plaque hippocampus,
    explicit "brain insulin resistance" findings (SGK1 down / IRS2 up). No GLP-1RA arm.
  - GSE262426: AppNL-G-F/hMapt AD knock-in mice x high-fat-diet/STZ insulin-resistance models.
  - GSE34451: Goto-Kakizaki (T2D) vs control rat hippocampus/PFC (microarray).
  - Any future GLP-1RA-treatment dataset, by changing GSE_ID and GROUP_COL/contrast below.

Pipeline stages:
  1. download_expression_matrix()   - GEOparse fetch + supplementary-file download,
                                       auto-detects counts-matrix vs series-matrix (microarray).
  2. run_differential_expression()  - PyDESeq2 for raw RNA-seq counts; Welch's t-test +
                                       FDR correction fallback for normalized/microarray data.
  3. run_pathway_enrichment()       - gseapy Enrichr against KEGG_2021_(Human|Mouse), Reactome.
  4. highlight_insulin_signaling_genes() - cross-references DE results against the KEGG
                                       Insulin signaling pathway (hsa04910/mmu04910) gene set,
                                       fetched live from the KEGG REST API with a static fallback.

Usage:
    pip install -r requirements.txt
    python geo_de_pathway_pipeline.py --gse GSE306976 --species mouse \
        --group-col genotype_stage --control "WT" --treatment "J20_24w"

Dry-run / mock mode (no network, no GEO/Enrichr/KEGG calls — for sanity-checking the
DE -> enrichment -> highlighting logic itself, e.g. in CI or offline):
    python geo_de_pathway_pipeline.py --dry-run
    python geo_de_pathway_pipeline.py --dry-run --dry-run-data-kind normalized
See also tests/test_pipeline.py for an automated pytest suite built on the same mock data.

Outputs (written to ./output/<GSE_ID or 'MOCK'>/):
    expression_matrix.csv, sample_metadata.csv, de_results.csv, enrichment_results.csv,
    insulin_signaling_genes_de.csv, volcano_plot.png, enrichment_barplot.png
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geo_de_pathway_pipeline")


# --------------------------------------------------------------------------- #
# 1. DOWNLOAD EXPRESSION MATRIX
# --------------------------------------------------------------------------- #

def download_expression_matrix(gse_id: str, dest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Fetch a GEO series and return (expression_matrix, sample_metadata, data_kind).

    expression_matrix : genes (rows) x samples (cols)
    sample_metadata    : samples (rows) x metadata fields (cols), indexed by sample (GSM) id
    data_kind          : "counts" (raw RNA-seq integer counts -> use PyDESeq2)
                          or "normalized" (microarray/already-normalized -> use t-test path)

    Strategy:
      a) Use GEOparse to pull the SOFT/series-matrix file for sample metadata (always available).
      b) Try to build the expression matrix from the embedded GSM table data
         (works for classic microarray series, e.g. GSE34451).
      c) If (b) yields nothing usable (common for RNA-seq series where expression
         values live in a separate supplementary file, e.g. GSE306976/GSE262426),
         download supplementary files and parse the first tabular file found
         (csv/tsv/txt/xlsx) into a genes x samples matrix.
    """
    import GEOparse  # local import: optional heavy dependency, only needed at runtime

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Fetching GEO series %s (this hits NCBI; cached locally after first run)", gse_id)
    gse = GEOparse.get_GEO(geo=gse_id, destdir=str(dest_dir), silent=True)

    # --- sample metadata -----------------------------------------------------
    meta_rows = {}
    for gsm_id, gsm in gse.gsms.items():
        meta_rows[gsm_id] = {k: "; ".join(v) if isinstance(v, list) else v
                              for k, v in gsm.metadata.items()}
    sample_metadata = pd.DataFrame.from_dict(meta_rows, orient="index")

    # --- (b) try embedded per-sample table data (microarray case) -----------
    expr_cols = {}
    for gsm_id, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is not None and not tbl.empty and {"ID_REF", "VALUE"}.issubset(tbl.columns):
            expr_cols[gsm_id] = tbl.set_index("ID_REF")["VALUE"]

    if expr_cols and len(expr_cols) == len(gse.gsms):
        expr = pd.DataFrame(expr_cols)
        expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        log.info("Built expression matrix from embedded GSM tables: %s genes x %s samples",
                  *expr.shape)
        return expr, sample_metadata, "normalized"

    # --- (c) fall back to supplementary files (RNA-seq counts case) ---------
    log.info("No usable embedded table data; downloading supplementary files for %s", gse_id)
    supp_dir = dest_dir / f"{gse_id}_supp"
    supp_dir.mkdir(exist_ok=True)
    gse.download_supplementary_files(directory=str(supp_dir), download_sra=False)

    candidate_files = sorted(
        p for p in supp_dir.rglob("*")
        if p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx"} and p.stat().st_size > 0
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"No parseable supplementary expression file found for {gse_id} in {supp_dir}. "
            "Inspect the GEO record manually (some series only deposit raw FASTQ/SRA, "
            "which requires an alignment/quantification pipeline outside this script's scope)."
        )

    # Heuristic: pick the largest candidate file (usually the full counts matrix,
    # as opposed to small per-sample peak/QC files).
    chosen = max(candidate_files, key=lambda p: p.stat().st_size)
    log.info("Parsing supplementary expression file: %s", chosen.name)
    sep = "\t" if chosen.suffix.lower() in {".tsv", ".txt"} else ","
    if chosen.suffix.lower() == ".xlsx":
        expr = pd.read_excel(chosen, index_col=0)
    else:
        expr = pd.read_csv(chosen, sep=sep, index_col=0)

    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    is_counts = np.allclose(expr.dropna().values, np.round(expr.dropna().values), atol=1e-6) \
        and expr.values.min() >= 0
    data_kind = "counts" if is_counts else "normalized"
    log.info("Parsed matrix: %s genes x %s samples (detected as %s)", *expr.shape, data_kind)
    return expr, sample_metadata, data_kind


# --------------------------------------------------------------------------- #
# 1b. MOCK DATA (DRY-RUN MODE) — no network, fully deterministic given `seed`
# --------------------------------------------------------------------------- #

def generate_mock_dataset(
    species: str = "mouse",
    data_kind: str = "counts",
    n_decoy_genes: int = 300,
    n_samples_per_group: int = 6,
    n_true_de_insulin_genes: int = 6,
    n_true_de_decoy_genes: int = 15,
    effect_size: float = 1.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build a synthetic genes x samples expression matrix + sample metadata, standing
    in for download_expression_matrix() so the rest of the pipeline (DE, enrichment,
    insulin-gene highlighting, plotting) can be exercised with zero network calls.

    Gene universe = the full KEGG insulin-signaling fallback gene set (so
    highlight_insulin_signaling_genes() has real signal to find) + n_decoy_genes
    unrelated genes. A known subset of each is made "truly" differentially
    expressed (log2 fold change = +/- effect_size, randomly signed) between two
    groups ("control" / "treatment"), the rest are noise around a flat baseline.

    Returns (expr, sample_metadata, ground_truth) where ground_truth is a dict
    with the gene lists that were deliberately perturbed, useful for asserting
    recall in tests (see tests/test_pipeline.py).
    """
    rng = np.random.default_rng(seed)

    insulin_genes = fetch_kegg_pathway_genes(
        "hsa04910" if species == "human" else "mmu04910", mock=True
    )
    decoy_genes = [f"Decoy{i:04d}" if species == "mouse" else f"DECOY{i:04d}"
                   for i in range(n_decoy_genes)]
    all_genes = insulin_genes + decoy_genes

    true_de_insulin = list(rng.choice(insulin_genes, size=min(n_true_de_insulin_genes, len(insulin_genes)),
                                       replace=False))
    true_de_decoy = list(rng.choice(decoy_genes, size=min(n_true_de_decoy_genes, len(decoy_genes)),
                                     replace=False))
    true_de_genes = set(true_de_insulin) | set(true_de_decoy)

    n_samples = 2 * n_samples_per_group
    sample_ids = [f"MOCK_GSM{i+1}" for i in range(n_samples)]
    groups = ["control"] * n_samples_per_group + ["treatment"] * n_samples_per_group

    baseline_mean = rng.uniform(50, 500, size=len(all_genes))  # per-gene baseline expression level
    fc_direction = rng.choice([-1, 1], size=len(all_genes))
    log2fc_true = np.array([
        fc_direction[i] * effect_size if gene in true_de_genes else 0.0
        for i, gene in enumerate(all_genes)
    ])

    expr = pd.DataFrame(index=all_genes, columns=sample_ids, dtype=float)
    for sample_id, group in zip(sample_ids, groups):
        group_mult = 2 ** log2fc_true if group == "treatment" else np.ones(len(all_genes))
        means = baseline_mean * group_mult
        if data_kind == "counts":
            # negative-binomial-like overdispersed counts via Poisson w/ gamma-distributed mean
            dispersed_means = rng.gamma(shape=10.0, scale=means / 10.0)
            expr[sample_id] = rng.poisson(lam=np.clip(dispersed_means, 1e-6, None))
        else:
            # log-normal-ish continuous "normalized" intensities (e.g. microarray-like)
            expr[sample_id] = rng.normal(loc=np.log2(means + 1), scale=0.3)

    sample_metadata = pd.DataFrame({"group": groups}, index=sample_ids)

    ground_truth = {
        "insulin_genes_in_universe": insulin_genes,
        "decoy_genes_in_universe": decoy_genes,
        "true_de_insulin_genes": true_de_insulin,
        "true_de_decoy_genes": true_de_decoy,
    }
    log.info("Generated mock dataset: %d genes x %d samples (%s), %d truly-DE insulin genes, "
             "%d truly-DE decoy genes", len(all_genes), n_samples, data_kind,
             len(true_de_insulin), len(true_de_decoy))
    return expr, sample_metadata, ground_truth


# --------------------------------------------------------------------------- #
# 2. DIFFERENTIAL EXPRESSION
# --------------------------------------------------------------------------- #

def run_differential_expression(
    expr: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    group_col: str,
    control: str,
    treatment: str,
    data_kind: str,
) -> pd.DataFrame:
    """
    Returns a DE results dataframe indexed by gene with columns:
    log2FoldChange, pvalue, padj  (column names match PyDESeq2 convention either way).
    """
    samples = sample_metadata.index[sample_metadata[group_col].isin([control, treatment])]
    expr_sub = expr.loc[:, expr.columns.intersection(samples)]
    groups = sample_metadata.loc[expr_sub.columns, group_col]

    if len(groups.unique()) != 2:
        raise ValueError(f"Expected exactly 2 groups for contrast, found: {groups.unique()}")

    if data_kind == "counts":
        log.info("Running PyDESeq2 (raw counts) for contrast %s vs %s", treatment, control)
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        counts_t = expr_sub.T.round().astype(int)  # samples x genes, required by PyDESeq2
        metadata = pd.DataFrame({group_col: groups})
        # drop genes with all-zero counts (DESeq2 cannot estimate dispersion otherwise)
        counts_t = counts_t.loc[:, (counts_t.sum(axis=0) > 0)]

        dds = DeseqDataSet(counts=counts_t, metadata=metadata, design_factors=group_col,
                            ref_level=[group_col, control])
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[group_col, treatment, control])
        stats.summary()
        results = stats.results_df.rename(columns={"log2FoldChange": "log2FoldChange",
                                                     "pvalue": "pvalue", "padj": "padj"})
        return results.sort_values("padj")

    # --- normalized/microarray fallback: Welch's t-test + BH-FDR -------------
    log.info("Running Welch's t-test + BH-FDR fallback for contrast %s vs %s", treatment, control)
    from scipy import stats as sstats
    from statsmodels.stats.multitest import multipletests

    a = expr_sub.loc[:, groups[groups == control].index]
    b = expr_sub.loc[:, groups[groups == treatment].index]
    a, b = a.dropna(), b.dropna()
    common_genes = a.index.intersection(b.index)
    a, b = a.loc[common_genes], b.loc[common_genes]

    tvals, pvals = sstats.ttest_ind(b.values, a.values, axis=1, equal_var=False, nan_policy="omit")
    log2fc = np.log2(b.mean(axis=1).clip(lower=1e-9) / a.mean(axis=1).clip(lower=1e-9))
    padj = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")[1]

    results = pd.DataFrame(
        {"log2FoldChange": log2fc.values, "stat": tvals, "pvalue": pvals, "padj": padj},
        index=common_genes,
    )
    return results.sort_values("padj")


# --------------------------------------------------------------------------- #
# 3. PATHWAY ENRICHMENT
# --------------------------------------------------------------------------- #

def run_pathway_enrichment(
    de_results: pd.DataFrame,
    species: str = "human",
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.5,
    gene_sets: list[str] | None = None,
    outdir: Path | None = None,
) -> pd.DataFrame:
    """
    Runs Enrichr-based pathway enrichment (via gseapy) on the set of significant
    genes from de_results. Requires internet access (queries the Enrichr web API).
    """
    import gseapy as gp

    gene_sets = gene_sets or (
        ["KEGG_2021_Human", "Reactome_2022"] if species == "human"
        else ["KEGG_2019_Mouse", "Reactome_2022"]
    )

    sig = de_results[(de_results["padj"] < padj_cutoff)
                      & (de_results["log2FoldChange"].abs() > log2fc_cutoff)]
    gene_list = [str(g).split(".")[0] for g in sig.index.tolist()]  # strip versioned IDs if any
    log.info("Running enrichment on %d significant genes (padj<%.2f, |log2FC|>%.2f) against %s",
              len(gene_list), padj_cutoff, log2fc_cutoff, gene_sets)

    if len(gene_list) < 5:
        log.warning("Fewer than 5 significant genes; enrichment results will be unreliable.")

    enr = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets, outdir=None, no_plot=True)
    results = enr.results.sort_values("Adjusted P-value")

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        results.to_csv(outdir / "enrichment_results.csv", index=False)

    # Flag any pathway whose name suggests insulin/IGF/AMPK/mTOR signaling, since
    # that's the mechanistic axis this whole workspace is testing.
    insulin_pattern = re.compile(r"insulin|igf-?1|ampk|mtor|pi3k.?akt", re.IGNORECASE)
    results["is_insulin_related_pathway"] = results["Term"].str.contains(insulin_pattern)
    return results


# --------------------------------------------------------------------------- #
# 3b. OFFLINE PATHWAY ENRICHMENT (DRY-RUN MODE) — no network, Fisher's exact test
# --------------------------------------------------------------------------- #

def _mock_gene_sets(species: str = "mouse") -> dict[str, list[str]]:
    """A tiny built-in 'gene set library' standing in for Enrichr/KEGG, so dry-run
    mode can exercise the full enrichment code path with zero network calls. One
    real-ish insulin-signaling set (the same fallback list used elsewhere) plus
    two unrelated decoy sets, so a correctly-working pipeline should flag the
    insulin set and *not* the decoys as significantly enriched."""
    insulin_genes = fetch_kegg_pathway_genes(
        "hsa04910" if species == "human" else "mmu04910", mock=True
    )
    rng = np.random.default_rng(7)
    decoy_a = [f"Decoy{i:04d}" if species == "mouse" else f"DECOY{i:04d}"
               for i in rng.choice(300, size=20, replace=False)]
    decoy_b = [f"Decoy{i:04d}" if species == "mouse" else f"DECOY{i:04d}"
               for i in rng.choice(300, size=20, replace=False)]
    return {
        "Insulin signaling pathway (mock-KEGG)": insulin_genes,
        "Unrelated decoy pathway A (mock)": decoy_a,
        "Unrelated decoy pathway B (mock)": decoy_b,
    }


def run_pathway_enrichment_offline(
    de_results: pd.DataFrame,
    species: str = "mouse",
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.5,
    gene_sets: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Network-free stand-in for run_pathway_enrichment(): computes enrichment of the
    DE significant gene list against a small local gene-set library via Fisher's
    exact test + BH-FDR, instead of querying the Enrichr web API. Output columns
    match run_pathway_enrichment()'s schema (Term, Adjusted P-value, Genes,
    is_insulin_related_pathway) so plot_enrichment_barplot() works unchanged.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    gene_sets = gene_sets or _mock_gene_sets(species)
    background = set(str(g) for g in de_results.index)
    sig_genes = set(de_results.index[
        (de_results["padj"] < padj_cutoff) & (de_results["log2FoldChange"].abs() > log2fc_cutoff)
    ].astype(str))

    rows = []
    for term, members in gene_sets.items():
        members = set(members) & background  # only genes actually present in this dataset
        if not members:
            continue
        overlap = sig_genes & members
        a = len(overlap)
        b = len(sig_genes - members)
        c = len(members - sig_genes)
        d = len(background - sig_genes - members)
        _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"Term": term, "Overlap": f"{a}/{len(members)}",
                      "Genes": ";".join(sorted(overlap)), "P-value": pvalue})

    results = pd.DataFrame(rows)
    if results.empty:
        log.warning("No gene-set members overlapped the dataset background; returning empty results.")
        return results
    results["Adjusted P-value"] = multipletests(results["P-value"], method="fdr_bh")[1]
    results = results.sort_values("Adjusted P-value").reset_index(drop=True)

    insulin_pattern = re.compile(r"insulin|igf-?1|ampk|mtor|pi3k.?akt", re.IGNORECASE)
    results["is_insulin_related_pathway"] = results["Term"].str.contains(insulin_pattern)
    return results


# --------------------------------------------------------------------------- #
# 4. HIGHLIGHT INSULIN SIGNALING GENES
# --------------------------------------------------------------------------- #

# Static fallback gene set (KEGG hsa04910 "Insulin signaling pathway" core members,
# human symbols). Used if the live KEGG REST fetch fails (offline / rate-limited).
FALLBACK_INSULIN_SIGNALING_GENES_HUMAN = [
    "INS", "INSR", "IRS1", "IRS2", "IRS4", "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2",
    "PDPK1", "AKT1", "AKT2", "AKT3", "GSK3B", "MTOR", "RPS6KB1", "EIF4EBP1",
    "FOXO1", "PRKAA1", "PRKAA2", "PRKAB1", "PRKAG1", "SLC2A4", "PYGL", "GYS1",
    "PPP1CA", "PPP1R3A", "SOS1", "GRB2", "RAF1", "MAP2K1", "MAPK1", "MAPK3",
    "RHEB", "TSC1", "TSC2", "PRKCZ", "CRK", "CRKL", "SORBS1", "FBP1", "PCK1",
    "G6PC1", "PHKA1", "PHKB", "PHKG1", "PYGM", "FASN", "SREBF1",
]

# Mouse orthologs are simply title-case of the human symbols for most of this set
# (e.g. INSR -> Insr); a few diverge and are listed explicitly.
_HUMAN_TO_MOUSE_OVERRIDES = {"G6PC1": "G6pc", "SLC2A4": "Slc2a4"}


def fetch_kegg_pathway_genes(pathway_id: str = "hsa04910", mock: bool = False) -> list[str]:
    """
    Fetch the gene list for a KEGG pathway (default: Insulin signaling pathway,
    human hsa04910; use 'mmu04910' for mouse) directly from the KEGG REST API.
    Falls back to a static curated list on any network/parsing failure, or
    immediately if mock=True (used by dry-run mode / tests to avoid any network call).
    """
    if mock:
        log.info("mock=True: skipping KEGG network call, using static fallback list for %s",
                  pathway_id)
        if pathway_id.startswith("mmu"):
            return [_HUMAN_TO_MOUSE_OVERRIDES.get(g, g.capitalize())
                    for g in FALLBACK_INSULIN_SIGNALING_GENES_HUMAN]
        return list(FALLBACK_INSULIN_SIGNALING_GENES_HUMAN)

    try:
        resp = requests.get(f"https://rest.kegg.jp/get/{pathway_id}", timeout=15)
        resp.raise_for_status()
        text = resp.text

        gene_section = re.search(r"^GENE\s+(.*?)(?=^\S)", text, re.DOTALL | re.MULTILINE)
        block = gene_section.group(1) if gene_section else ""
        # Each gene line looks like: "3630  INS; insulin [KO:K04524]"
        symbols = re.findall(r"^\s*\d+\s+([A-Za-z0-9_\-\.]+);", block, re.MULTILINE)
        if symbols:
            log.info("Fetched %d genes for KEGG pathway %s", len(symbols), pathway_id)
            return symbols
        log.warning("KEGG response for %s parsed to 0 genes; using fallback list", pathway_id)
    except Exception as exc:  # noqa: BLE001 - any network/parse failure -> fallback
        log.warning("KEGG fetch failed (%s); using fallback list", exc)

    if pathway_id.startswith("mmu"):
        return [_HUMAN_TO_MOUSE_OVERRIDES.get(g, g.capitalize())
                for g in FALLBACK_INSULIN_SIGNALING_GENES_HUMAN]
    return FALLBACK_INSULIN_SIGNALING_GENES_HUMAN


def highlight_insulin_signaling_genes(
    de_results: pd.DataFrame, species: str = "human", mock: bool = False
) -> pd.DataFrame:
    """
    Cross-reference DE results against the KEGG insulin signaling pathway gene set.
    Returns the subset of de_results whose gene index matches that set, with an
    added 'in_insulin_signaling_pathway' column on the *full* results too (in place).

    mock=True skips the live KEGG REST call (used by --dry-run and the test suite).
    """
    pathway_id = "hsa04910" if species == "human" else "mmu04910"
    insulin_genes = set(fetch_kegg_pathway_genes(pathway_id, mock=mock))

    # Case-insensitive matching since mouse symbols are title-case and some
    # GEO platforms use all-caps probe annotations regardless of species.
    index_upper = {str(g).upper(): g for g in de_results.index}
    insulin_genes_upper = {g.upper() for g in insulin_genes}
    matched_original = [orig for upper, orig in index_upper.items() if upper in insulin_genes_upper]

    de_results["in_insulin_signaling_pathway"] = de_results.index.isin(matched_original)
    subset = de_results.loc[matched_original].sort_values("padj")
    log.info("%d/%d insulin-signaling-pathway genes found in this dataset; %d are DE (padj<0.05)",
              len(matched_original), len(insulin_genes),
              (subset["padj"] < 0.05).sum() if len(subset) else 0)
    return subset


# --------------------------------------------------------------------------- #
# VISUALIZATION
# --------------------------------------------------------------------------- #

def plot_volcano(de_results: pd.DataFrame, outpath: Path,
                  padj_cutoff: float = 0.05, log2fc_cutoff: float = 0.5) -> None:
    import matplotlib.pyplot as plt

    df = de_results.copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    sig = (df["padj"] < padj_cutoff) & (df["log2FoldChange"].abs() > log2fc_cutoff)
    insulin = df.get("in_insulin_signaling_pathway", pd.Series(False, index=df.index))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[~sig & ~insulin, "log2FoldChange"], df.loc[~sig & ~insulin, "neg_log10_padj"],
               s=8, color="lightgrey", label="Not significant", alpha=0.6)
    ax.scatter(df.loc[sig & ~insulin, "log2FoldChange"], df.loc[sig & ~insulin, "neg_log10_padj"],
               s=10, color="steelblue", label="Significant (DE)", alpha=0.7)
    ax.scatter(df.loc[insulin, "log2FoldChange"], df.loc[insulin, "neg_log10_padj"],
               s=40, color="crimson", edgecolor="black", linewidth=0.4,
               label="Insulin signaling pathway (KEGG)", zorder=5)

    for gene in df.index[insulin & sig]:
        ax.annotate(str(gene), (df.loc[gene, "log2FoldChange"], df.loc[gene, "neg_log10_padj"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.axhline(-np.log10(padj_cutoff), color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(log2fc_cutoff, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(-log2fc_cutoff, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Differential expression — insulin signaling genes highlighted")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved volcano plot to %s", outpath)


def plot_enrichment_barplot(enrichment_results: pd.DataFrame, outpath: Path, top_n: int = 15) -> None:
    import matplotlib.pyplot as plt

    top = enrichment_results.sort_values("Adjusted P-value").head(top_n).iloc[::-1]
    colors = ["crimson" if flag else "steelblue" for flag in top["is_insulin_related_pathway"]]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(top["Term"], -np.log10(top["Adjusted P-value"].clip(lower=1e-300)), color=colors)
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(f"Top {top_n} enriched pathways (red = insulin/IGF/AMPK/mTOR-related)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log.info("Saved enrichment barplot to %s", outpath)


# --------------------------------------------------------------------------- #
# DRIVER
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gse", default="GSE306976",
                         help="GEO series accession (default: GSE306976)")
    parser.add_argument("--species", choices=["human", "mouse"], default="mouse")
    parser.add_argument("--group-col", default="characteristics_ch1",
                         help="Column in sample metadata that distinguishes groups "
                              "(inspect sample_metadata.csv after a first run to find the right one)")
    parser.add_argument("--control", default="control", help="Control group label")
    parser.add_argument("--treatment", default="treatment", help="Treatment/comparison group label")
    parser.add_argument("--padj", type=float, default=0.05)
    parser.add_argument("--log2fc", type=float, default=0.5)
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run end-to-end on synthetic data with zero network calls "
                              "(no GEO/Enrichr/KEGG access). Ignores --gse/--group-col/"
                              "--control/--treatment, which are fixed to the mock dataset's "
                              "own labels ('group' / 'control' / 'treatment').")
    parser.add_argument("--dry-run-data-kind", choices=["counts", "normalized"], default="counts",
                         help="Which run_differential_expression() code path to exercise in "
                              "dry-run mode: 'counts' -> PyDESeq2, 'normalized' -> t-test+FDR.")
    parser.add_argument("--dry-run-seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        log.warning("=== DRY-RUN MODE: using synthetic data, no network calls will be made ===")
        gse_label = "MOCK"
        outdir = Path(args.outdir) / gse_label
        outdir.mkdir(parents=True, exist_ok=True)

        expr, sample_metadata, ground_truth = generate_mock_dataset(
            species=args.species, data_kind=args.dry_run_data_kind, seed=args.dry_run_seed
        )
        data_kind = args.dry_run_data_kind
        group_col, control, treatment = "group", "control", "treatment"
    else:
        outdir = Path(args.outdir) / args.gse
        outdir.mkdir(parents=True, exist_ok=True)

        expr, sample_metadata, data_kind = download_expression_matrix(args.gse, outdir / "raw")
        group_col, control, treatment = args.group_col, args.control, args.treatment
        ground_truth = None

    expr.to_csv(outdir / "expression_matrix.csv")
    sample_metadata.to_csv(outdir / "sample_metadata.csv")

    de_results = run_differential_expression(expr, sample_metadata, group_col, control, treatment, data_kind)

    insulin_subset = highlight_insulin_signaling_genes(
        de_results, species=args.species, mock=args.dry_run
    )
    de_results.to_csv(outdir / "de_results.csv")
    insulin_subset.to_csv(outdir / "insulin_signaling_genes_de.csv")

    if args.dry_run:
        enrichment = run_pathway_enrichment_offline(
            de_results, species=args.species, padj_cutoff=args.padj, log2fc_cutoff=args.log2fc
        )
        if not enrichment.empty:
            enrichment.to_csv(outdir / "enrichment_results.csv", index=False)
    else:
        enrichment = run_pathway_enrichment(
            de_results, species=args.species, padj_cutoff=args.padj,
            log2fc_cutoff=args.log2fc, outdir=outdir,
        )

    plot_volcano(de_results, outdir / "volcano_plot.png", args.padj, args.log2fc)
    if not enrichment.empty:
        plot_enrichment_barplot(enrichment, outdir / "enrichment_barplot.png")

    log.info("Done. Results written to %s", outdir.resolve())
    log.info("Insulin-signaling-pathway genes found in dataset: %d (DE at padj<%.2f: %d)",
              len(insulin_subset), args.padj, (insulin_subset["padj"] < args.padj).sum())

    if ground_truth is not None:
        detected_insulin_de = set(insulin_subset.index[insulin_subset["padj"] < args.padj])
        true_insulin_de = set(ground_truth["true_de_insulin_genes"])
        recall = len(detected_insulin_de & true_insulin_de) / max(len(true_insulin_de), 1)
        log.info("Dry-run sanity check: recovered %d/%d deliberately-perturbed insulin genes "
                  "(recall=%.2f). False positives among insulin genes: %s",
                  len(detected_insulin_de & true_insulin_de), len(true_insulin_de), recall,
                  sorted(detected_insulin_de - true_insulin_de) or "none")


if __name__ == "__main__":
    main()
