"""
glp1_model.py — unified model for testing central insulin resistance as a mediator
of GLP-1 receptor agonist neuroprotection in Alzheimer's and Parkinson's disease.

This merges what were two separate scripts (geo_de_pathway_pipeline.py and
mediation_model.py) into one module with one CLI, organized as three stages of
the same overall question:

  STAGE A — GEO TRANSCRIPTOMICS  (download -> differential expression -> pathway
            enrichment -> insulin-signaling-gene highlighting)
            Tests the "is central insulin resistance present / does it change"
            side of the hypothesis, at the gene-expression level.

  STAGE B — CAUSAL MEDIATION MODEL  (Baron-Kenny path model + bootstrap CI)
            Tests the "does a treatment -> insulin-marker -> outcome causal
            chain actually exist" side of the hypothesis, at the patient level.

  STAGE C — INTEGRATED PIPELINE DEMO
            Shows how Stage A's output WOULD feed Stage B's input if a single
            dataset ever linked GLP-1RA treatment, an insulin-signaling marker,
            and a neurodegenerative outcome in the same subjects. No such public
            dataset currently exists (confirmed across this project's full
            literature + GEO search — see ../extraction/literature_extraction.json
            and ../extraction/contradicting_evidence.json), so this stage runs on
            each half's own mock/simulated data and is illustrative plumbing, not
            a real combined finding.

WHAT THIS MODEL CAN AND CANNOT DO — read before treating any output as evidence:
  - Stage A can run on REAL GEO data (no GLP-1RA-treated AD/PD dataset exists yet,
    but background insulin-resistance-in-AD/PD datasets do — see module docstring
    of run_geo() below for the specific accessions this was designed against).
  - Stage B can run on REAL patient-level data the moment it exists, via `--csv`.
  - Until then, both stages' "interesting" outputs are dry-run/illustrative and
    are labeled as such everywhere (in logs, in print output, and in this docstring).

Usage:
    pip install -r requirements.txt

    # Stage A: GEO pipeline
    python glp1_model.py geo --dry-run
    python glp1_model.py geo --gse GSE306976 --species mouse \
        --group-col genotype_stage --control "WT" --treatment "J20_24w"

    # Stage B: mediation model
    python glp1_model.py mediation --dry-run
    python glp1_model.py mediation --simulate-illustration
    python glp1_model.py mediation --power-analysis --true-a 0.27 --true-b 0.5 --true-cprime 1.75
    python glp1_model.py mediation --csv ipd.csv --treatment drug --mediator irs1_change --outcome updrs_change

    # Stage C: integrated demo (mock GEO output feeding the mediation model)
    python glp1_model.py pipeline --dry-run

See analysis/tests/test_glp1_model.py for the full offline pytest suite (no
network calls anywhere in the test suite).
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("glp1_model")


# =========================================================================== #
# STAGE A — GEO TRANSCRIPTOMICS PIPELINE
# =========================================================================== #
#
# Context (see ../extraction/literature_extraction.json -> additional_database_sources.GEO_transcriptomics):
# No GEO dataset currently combines AD + insulin signaling + GLP-1 pathway treatment
# in one experiment. This pipeline is written generically so it can be pointed at:
#   - GSE306976 (default): J20 AD mouse model, pre- vs post-amyloid-plaque hippocampus,
#     explicit "brain insulin resistance" findings (SGK1 down / IRS2 up). No GLP-1RA arm.
#   - GSE262426: AppNL-G-F/hMapt AD knock-in mice x high-fat-diet/STZ insulin-resistance models.
#   - GSE34451: Goto-Kakizaki (T2D) vs control rat hippocampus/PFC (microarray).
#   - Any future GLP-1RA-treatment dataset, by changing GSE_ID and GROUP_COL/contrast below.

# --------------------------------------------------------------------------- #
# A1. DOWNLOAD EXPRESSION MATRIX
# --------------------------------------------------------------------------- #

def _strip_known_compression(p: Path) -> str:
    """Lowercase filename with a trailing .gz/.bz2/.zip/.xz stripped, for extension
    sniffing on compressed supplementary files (e.g. 'counts.csv.gz' -> 'counts.csv')."""
    name = p.name.lower()
    for comp_ext in (".gz", ".bz2", ".zip", ".xz"):
        if name.endswith(comp_ext):
            return name[: -len(comp_ext)]
    return name


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
         (works for classic microarray series with a Series Matrix File, e.g. GSE34451 —
         this is the most reliable path and needs no supplementary-file parsing at all).
      c) If (b) yields nothing usable (common for RNA-seq series where expression
         values live in a separate supplementary file), download per-SAMPLE (GSM-level)
         supplementary files AND per-SERIES (GSE-level) supplementary files (GEOparse's
         download_supplementary_files() only fetches the former; many series, e.g.
         GSE262426/GSE306976, attach their actual data file to the Series record instead),
         extract any .tar/.tar.gz archives found, then parse the first tabular file
         (csv/tsv/txt/xlsx, optionally .gz-compressed) into a genes x samples matrix.

    Known caveat: not every series' supplementary file is actually a gene-level
    expression/count matrix even when it's a parseable table — e.g. GSE306976's
    only supplementary file is 'GSE306976_Read_Quantification_Summary_stat.xlsx',
    which is per-sample QC/alignment statistics, not gene counts. This function
    cannot detect that semantic mismatch automatically; inspect the parsed
    expression_matrix.csv's row index after a run to confirm it looks like gene
    symbols/IDs, not QC metric names, before trusting downstream results.
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

    # (c1) GSM-level (per-sample) supplementary files, via GEOparse.
    gse.download_supplementary_files(directory=str(supp_dir), download_sra=False)

    # (c2) GSE-level (series-level) supplementary file(s). GEOparse's
    # download_supplementary_files() above only walks gse.gsms, so a series whose
    # data file is attached to the Series record itself (filename prefixed with
    # the GSE accession, not a GSM accession — e.g. GSE306976_Read_Quantification...)
    # is silently skipped unless we fetch it ourselves here.
    series_supp_urls = gse.metadata.get("supplementary_file", []) if hasattr(gse, "metadata") else []
    for url in series_supp_urls:
        fname = url.rstrip("/").split("/")[-1]
        dest_path = supp_dir / fname
        if dest_path.exists() and dest_path.stat().st_size > 0:
            continue
        log.info("Downloading series-level supplementary file: %s", fname)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001 - network failure on one URL shouldn't abort the whole run
            log.warning("Failed to download series-level supplementary file %s: %s", url, exc)

    # (c3) Extract any .tar/.tar.gz archives found so far (common for RNA-seq
    # series that bundle one file per sample, e.g. GSE262426_RAW.tar).
    import tarfile

    for archive in list(supp_dir.rglob("*.tar")) + list(supp_dir.rglob("*.tar.gz")):
        log.info("Extracting archive: %s", archive.name)
        try:
            with tarfile.open(archive) as tf:
                tf.extractall(path=supp_dir / archive.stem)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to extract %s: %s", archive.name, exc)

    tabular_extensions = (".csv", ".tsv", ".txt", ".xlsx")
    candidate_files = sorted(
        p for p in supp_dir.rglob("*")
        if p.is_file() and _strip_known_compression(p).endswith(tabular_extensions)
        and p.stat().st_size > 0
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"No parseable supplementary expression file found for {gse_id} in {supp_dir}. "
            "Inspect the GEO record manually (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
            f"acc={gse_id}) — some series only deposit raw FASTQ/SRA, which requires an "
            "alignment/quantification pipeline outside this script's scope, and some series' "
            "only supplementary file is QC/summary statistics rather than a gene-level matrix."
        )

    # Heuristic: pick the largest candidate file (usually the full counts matrix,
    # as opposed to small per-sample peak/QC files).
    chosen = max(candidate_files, key=lambda p: p.stat().st_size)
    log.info("Parsing supplementary expression file: %s", chosen.name)
    stripped_name = _strip_known_compression(chosen)
    sep = "\t" if stripped_name.endswith((".tsv", ".txt")) else ","
    if stripped_name.endswith(".xlsx"):
        expr = pd.read_excel(chosen, index_col=0)
    else:
        # compression="infer" (pandas default) auto-detects .gz/.bz2/.zip/.xz from the filename.
        expr = pd.read_csv(chosen, sep=sep, index_col=0)

    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    is_counts = np.allclose(expr.dropna().values, np.round(expr.dropna().values), atol=1e-6) \
        and expr.values.min() >= 0
    data_kind = "counts" if is_counts else "normalized"
    log.info("Parsed matrix: %s genes x %s samples (detected as %s)", *expr.shape, data_kind)
    return expr, sample_metadata, data_kind


# --------------------------------------------------------------------------- #
# A2. MOCK GEO DATA (DRY-RUN MODE) — no network, fully deterministic given `seed`
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
    recall in tests (see tests/test_glp1_model.py).
    """
    rng = np.random.default_rng(seed)

    insulin_genes = fetch_kegg_pathway_genes(
        _kegg_pathway_id(species), mock=True
    )
    decoy_genes = [f"Decoy{i:04d}" if species in TITLECASE_SPECIES else f"DECOY{i:04d}"
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
# A3. DIFFERENTIAL EXPRESSION
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
# A4. PATHWAY ENRICHMENT
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
# A5. OFFLINE PATHWAY ENRICHMENT (DRY-RUN MODE) — no network, Fisher's exact test
# --------------------------------------------------------------------------- #

def _mock_gene_sets(species: str = "mouse") -> dict[str, list[str]]:
    """A tiny built-in 'gene set library' standing in for Enrichr/KEGG, so dry-run
    mode can exercise the full enrichment code path with zero network calls. One
    real-ish insulin-signaling set (the same fallback list used elsewhere) plus
    two unrelated decoy sets, so a correctly-working pipeline should flag the
    insulin set and *not* the decoys as significantly enriched."""
    insulin_genes = fetch_kegg_pathway_genes(
        _kegg_pathway_id(species), mock=True
    )
    rng = np.random.default_rng(7)
    decoy_a = [f"Decoy{i:04d}" if species in TITLECASE_SPECIES else f"DECOY{i:04d}"
               for i in rng.choice(300, size=20, replace=False)]
    decoy_b = [f"Decoy{i:04d}" if species in TITLECASE_SPECIES else f"DECOY{i:04d}"
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
# A6. HIGHLIGHT INSULIN SIGNALING GENES
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

# Species support: human uses ALL-CAPS gene symbols; mouse and rat both use
# Title Case (e.g. Insr, Irs1) and share the same KEGG-pathway-id prefix scheme
# (mmuNNNNN / rnoNNNNN), so they're handled identically everywhere below.
SPECIES_TO_KEGG_PREFIX = {"human": "hsa", "mouse": "mmu", "rat": "rno"}
TITLECASE_SPECIES = {"mouse", "rat"}


def _kegg_pathway_id(species: str, pathway_number: str = "04910") -> str:
    """e.g. _kegg_pathway_id('rat') -> 'rno04910'."""
    prefix = SPECIES_TO_KEGG_PREFIX.get(species, "mmu")
    return f"{prefix}{pathway_number}"


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
        if pathway_id[:3] in ("mmu", "rno"):  # mouse, rat: both use Title Case symbols
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

    if pathway_id[:3] in ("mmu", "rno"):  # mouse, rat: both use Title Case symbols
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
    pathway_id = _kegg_pathway_id(species)
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
# A7. GEO VISUALIZATION
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
# A8. STAGE A DRIVER
# --------------------------------------------------------------------------- #

def run_geo(args: argparse.Namespace) -> dict:
    """Executes the full Stage A pipeline for the `geo` subcommand. Returns a
    dict of the key in-memory artifacts (de_results, insulin_subset, enrichment,
    ground_truth) so Stage C (the integrated demo) can reuse them directly."""
    if args.dry_run:
        log.warning("=== DRY-RUN MODE: using synthetic data, no network calls will be made ===")
        outdir = Path(args.outdir) / "MOCK"
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

    return {"de_results": de_results, "insulin_subset": insulin_subset,
            "enrichment": enrichment, "ground_truth": ground_truth}


# =========================================================================== #
# STAGE B — CAUSAL MEDIATION MODEL
# =========================================================================== #
#
# Tests: GLP-1RA treatment (X) --a--> insulin-resistance marker (M) --b--> outcome (Y)
#                              \\___________ c' (direct effect) ___________/
#
# This is the literal analytical step this whole project has found missing from
# every paper reviewed (literature_extraction.json -> synthesis_summary.key_finding:
# "No identified study ... performed a formal statistical mediation analysis").
#
# No public individual-patient-data (IPD) dataset exists that pairs GLP-1RA
# treatment, a quantified central-insulin-resistance marker, and a
# neurodegenerative outcome in the SAME subjects (confirmed across this
# project's full literature + GEO search). fit_mediation() is ready to run on
# real IPD the moment it exists; until then, simulate_athauda2019_illustration()
# provides a clearly-labeled calibrated illustration, and run_power_analysis()
# answers a real, evidence-adjacent question: were existing trials even large
# enough to detect mediation if it's real?

# --------------------------------------------------------------------------- #
# B1. CORE MEDIATION MODEL
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
        def sig(p: float) -> str:
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

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
# B2. SIMULATION — generic (no calibration claims)
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
# B3. SIMULATION — calibrated illustration of the Athauda et al. 2017/2019 scenario
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
# B4. POWER / FEASIBILITY ANALYSIS
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
# B5. MEDIATION VISUALIZATION
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
# B6. STAGE B DRIVER
# --------------------------------------------------------------------------- #

def run_mediation(args: argparse.Namespace) -> MediationResult | None:
    """Executes the full Stage B pipeline for the `mediation` subcommand."""
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
        return result

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
        return result

    elif args.power_analysis:
        n_list = [int(x) for x in args.n_list.split(",")]
        # Compute cost is roughly n_sims * len(n_list) * (1 + power_n_boot) * 3
        # regression fits — note this uses --power-n-boot, NOT --n-boot (the latter
        # is for a single final fit elsewhere and is deliberately NOT used here,
        # since nesting it inside n_sims x len(n_list) simulations would explode
        # runtime, e.g. the old default combination ran for ~30 min on a typical run).
        est_fits = args.n_sims * len(n_list) * (1 + args.power_n_boot) * 3
        log.info("Power analysis cost estimate: ~%d total regression fits "
                  "(n_sims=%d x %d sample sizes x (1+power_n_boot=%d) x 3 fits each)",
                  est_fits, args.n_sims, len(n_list), args.power_n_boot)
        power_df = run_power_analysis(
            n_list=n_list, true_a=args.true_a, true_b=args.true_b, true_cprime=args.true_cprime,
            n_sims=args.n_sims, n_boot=args.power_n_boot, seed=args.seed,
        )
        print(power_df.to_string(index=False))
        power_df.to_csv(outdir / "power_analysis.csv", index=False)
        plot_power_curve(power_df, outdir / "power_curve.png")
        return None

    elif args.dry_run:
        log.info("=== DRY RUN: arbitrary synthetic data, mechanism smoke-test only ===")
        df = simulate_mediation_dataset(n=300, true_a=1.0, true_b=0.8, true_cprime=0.3, seed=args.seed)
        result = fit_mediation(df, "treatment", "mediator", "outcome", n_boot=2000, seed=args.seed)
        print(result.summary())
        plot_path_diagram(result, "treatment", "mediator", "outcome", outdir / "dry_run_path_diagram.png")
        return result

    else:
        log.info("No mediation mode selected; pass --csv, --simulate-illustration, "
                 "--power-analysis, or --dry-run.")
        return None


# =========================================================================== #
# STAGE C — INTEGRATED PIPELINE DEMO
# =========================================================================== #

def run_pipeline_demo(args: argparse.Namespace) -> dict:
    """
    Demonstrates how Stage A's output WOULD feed Stage B's input if a single
    dataset ever linked GLP-1RA treatment, an insulin-signaling marker, and a
    neurodegenerative outcome in the same subjects.

    Mechanics (illustrative plumbing only — both halves still run on each
    module's own mock/simulated data internally, since no real linked dataset
    exists):
      1. Run Stage A's dry-run GEO pipeline -> get DE results + insulin-gene subset.
      2. Take the most significant insulin-signaling gene's |log2FoldChange| as a
         stand-in for the mediation model's "path a" (treatment -> mediator)
         effect size.
      3. Feed that into Stage B's generic simulator (simulate_mediation_dataset)
         alongside a user-specified assumed path b, and run the full mediation
         analysis on the result.

    This shows the conceptual handoff end-to-end; it is NOT a claim that a real
    gene's expression change was causally linked to a real clinical outcome.
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log.warning("=== PIPELINE DEMO: chaining Stage A's mock GEO output into Stage B. "
                "Illustrative plumbing only — see run_pipeline_demo() docstring. ===")

    geo_args = argparse.Namespace(
        dry_run=True, dry_run_data_kind="counts", dry_run_seed=args.seed,
        species=args.species, outdir=str(outdir / "geo_stage"),
        padj=0.05, log2fc=0.5,
    )
    geo_outputs = run_geo(geo_args)

    insulin_subset = geo_outputs["insulin_subset"]
    if insulin_subset.empty:
        log.warning("No significant insulin-signaling genes found in the mock GEO run; "
                     "using a small placeholder effect size for Stage B instead.")
        handoff_effect_size = 0.3
        handoff_gene = "(none significant)"
    else:
        top_gene = insulin_subset["padj"].idxmin()
        handoff_effect_size = float(abs(insulin_subset.loc[top_gene, "log2FoldChange"]))
        handoff_gene = str(top_gene)

    log.info("Stage A -> Stage B handoff: using |log2FC|=%.3f from gene '%s' as path-a effect size",
              handoff_effect_size, handoff_gene)

    med_df = simulate_mediation_dataset(
        n=args.mediation_n, true_a=handoff_effect_size, true_b=args.assumed_true_b,
        true_cprime=args.assumed_true_cprime, seed=args.seed,
    )
    result = fit_mediation(med_df, "treatment", "mediator", "outcome", n_boot=2000, seed=args.seed,
                            notes=[f"Path-a magnitude ({handoff_effect_size:.3f}) taken from mock GEO "
                                   f"gene '{handoff_gene}'; path-b ({args.assumed_true_b}) and total "
                                   f"effect (c'={args.assumed_true_cprime}) are user-assumed, not derived "
                                   f"from any real data. This entire demo runs on Stage A's and Stage B's "
                                   f"own mock/simulated data."])
    print(result.summary())
    plot_path_diagram(result, "GLP-1RA (mock)", f"{handoff_gene} expression (mock)",
                       "Neurodegenerative outcome (simulated)",
                       outdir / "pipeline_demo_path_diagram.png",
                       title="Stage A -> Stage B integrated demo (mock data throughout)")

    return {"geo_outputs": geo_outputs, "mediation_result": result,
            "handoff_gene": handoff_gene, "handoff_effect_size": handoff_effect_size}


# =========================================================================== #
# CLI DRIVER
# =========================================================================== #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glp1_model", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Stage A: geo ---
    geo_parser = subparsers.add_parser("geo", help="GEO download -> DE -> enrichment -> insulin-gene highlighting")
    geo_parser.add_argument("--gse", default="GSE306976", help="GEO series accession (default: GSE306976)")
    geo_parser.add_argument("--species", choices=["human", "mouse", "rat"], default="mouse")
    geo_parser.add_argument("--group-col", default="characteristics_ch1",
                             help="Column in sample metadata that distinguishes groups "
                                  "(inspect sample_metadata.csv after a first run to find the right one)")
    geo_parser.add_argument("--control", default="control", help="Control group label")
    geo_parser.add_argument("--treatment", default="treatment", help="Treatment/comparison group label")
    geo_parser.add_argument("--padj", type=float, default=0.05)
    geo_parser.add_argument("--log2fc", type=float, default=0.5)
    geo_parser.add_argument("--outdir", default="output")
    geo_parser.add_argument("--dry-run", action="store_true",
                             help="Run end-to-end on synthetic data with zero network calls")
    geo_parser.add_argument("--dry-run-data-kind", choices=["counts", "normalized"], default="counts")
    geo_parser.add_argument("--dry-run-seed", type=int, default=42)

    # --- Stage B: mediation ---
    med_parser = subparsers.add_parser("mediation", help="Causal mediation analysis")
    med_parser.add_argument("--csv", help="Path to real patient-level data (treatment/mediator/outcome columns)")
    med_parser.add_argument("--treatment", default="treatment")
    med_parser.add_argument("--mediator", default="mediator")
    med_parser.add_argument("--outcome", default="outcome")
    med_parser.add_argument("--covariates", default="", help="Comma-separated covariate column names")
    med_parser.add_argument("--n-boot", type=int, default=5000)
    med_parser.add_argument("--seed", type=int, default=42)
    med_parser.add_argument("--outdir", default="output/mediation")
    med_parser.add_argument("--simulate-illustration", action="store_true",
                             help="Run the calibrated-but-illustrative Athauda 2017/2019 scenario "
                                  "(NOT real data)")
    med_parser.add_argument("--assumed-proportion-mediated", type=float, default=0.4)
    med_parser.add_argument("--power-analysis", action="store_true")
    med_parser.add_argument("--true-a", type=float, default=0.27)
    med_parser.add_argument("--true-b", type=float, default=0.5)
    med_parser.add_argument("--true-cprime", type=float, default=1.75)
    med_parser.add_argument("--n-list", default="62,300,1200,3808",
                             help="Comma-separated sample sizes to evaluate. Default trimmed to 4 "
                                  "values (was 6) since cost scales linearly with this count.")
    med_parser.add_argument("--n-sims", type=int, default=100,
                             help="Simulated trials per sample size. Default lowered from 300 to 100 "
                                  "(see --power-n-boot note: cost is n_sims x len(n_list) x (1+power_n_boot) x 3).")
    med_parser.add_argument("--power-n-boot", type=int, default=200,
                             help="Bootstrap resamples PER SIMULATED TRIAL during --power-analysis "
                                  "(deliberately separate from --n-boot, which controls a single final "
                                  "fit's precision elsewhere). This value is nested inside n_sims x "
                                  "len(n_list) simulations, so it dominates runtime far more than "
                                  "--n-boot does for any other mode — keep this low (100-300 is typical "
                                  "for a binary CI-excludes-zero decision per simulation; raise only if "
                                  "you have time to spare).")
    med_parser.add_argument("--dry-run", action="store_true",
                             help="Quick offline smoke test on arbitrary simulated data")

    # --- Stage C: pipeline demo ---
    pipe_parser = subparsers.add_parser("pipeline", help="Integrated demo: mock GEO output feeds mediation model")
    pipe_parser.add_argument("--species", choices=["human", "mouse", "rat"], default="mouse")
    pipe_parser.add_argument("--seed", type=int, default=42)
    pipe_parser.add_argument("--outdir", default="output/pipeline")
    pipe_parser.add_argument("--mediation-n", type=int, default=200,
                              help="Sample size for the Stage B simulation in this demo")
    pipe_parser.add_argument("--assumed-true-b", type=float, default=0.8)
    pipe_parser.add_argument("--assumed-true-cprime", type=float, default=0.3)
    pipe_parser.add_argument("--dry-run", action="store_true",
                              help="Present for interface consistency; pipeline mode is always mock-data-based")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "geo":
        run_geo(args)
    elif args.command == "mediation":
        run_mediation(args)
    elif args.command == "pipeline":
        run_pipeline_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
