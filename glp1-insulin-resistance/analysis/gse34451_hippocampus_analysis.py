"""
GSE34451 Hippocampus, Control vs T2D (Goto-Kakizaki) — full analysis + visualization.

Consolidates the real, working analysis developed interactively in a Colab session
(see chat history / research_question.md for the full narrative) into one
reproducible script. Produces every figure/table referenced in the Results
section of the research note.

Dataset: Abdul-Rahman et al., GSE34451. Agilent rat whole-genome microarray
(platform GPL15011), 27 samples = 3 brain regions (Hippocampus, Prefrontal
Cortex, Striatum) x 3 groups (Control/Wistar, T1D/streptozotocin-treated,
T2D/Goto-Kakizaki) x 3 replicates. This script analyzes the Hippocampus,
Control-vs-T2D contrast specifically (most directly relevant to the "type 3
diabetes" / central insulin resistance hypothesis); change REGION/CONTROL/
TREATMENT below to analyze a different contrast (e.g. Hippocampus T1D, or a
different brain region).

Known limitation to state explicitly in any write-up: n=3 per group per
region. This is the real, actual sample size of the public dataset, not a
choice made by this script — expect limited statistical power and treat any
finding here as exploratory/hypothesis-generating, not confirmatory.

Usage (Colab or local):
    pip install -r requirements.txt
    python gse34451_hippocampus_analysis.py

Outputs (written to ./output/GSE34451_hippocampus_T2D/):
    de_results.csv, insulin_signaling_genes_de.csv, expr_matrix_annotated.csv,
    volcano_plot.png, insulin_genes_heatmap.png, top_insulin_genes_boxplots.png,
    pca_plot.png, pvalue_histogram.png, sample_correlation_heatmap.png,
    insulin_genes_forestplot.png, enrichment_results.csv (if Enrichr reachable),
    enrichment_barplot.png (if enrichment found any hits), summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glp1_model as gm  # noqa: E402

# --------------------------------------------------------------------------- #
# CONFIG — change these to analyze a different region/contrast
# --------------------------------------------------------------------------- #
GSE_ID = "GSE34451"
GPL_ID = "GPL15011"
REGION = "Hippocampus"      # one of: Hippocampus, PrefrontalCortex, Striatum
CONTROL_GROUP = "Control"   # one of: Control, T1D, T2D
TREATMENT_GROUP = "T2D"     # one of: Control, T1D, T2D
SPECIES = "rat"
OUTDIR = Path("output") / f"{GSE_ID}_{REGION.lower()}_{TREATMENT_GROUP.lower()}"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    import GEOparse

    # --- 1. Download series + platform annotation -------------------------
    gm.log.info("Downloading %s expression data...", GSE_ID)
    expr, sample_metadata, data_kind = gm.download_expression_matrix(GSE_ID, OUTDIR.parent / f"{GSE_ID}_raw")

    gm.log.info("Downloading %s platform annotation...", GPL_ID)
    gpl = GEOparse.get_GEO(geo=GPL_ID, destdir=str(OUTDIR.parent / f"{GSE_ID}_raw"), silent=True)

    # --- 2. Map probe IDs -> gene symbols, collapse multi-probe genes -----
    id_to_symbol = gpl.table.set_index("ID")["GENE_SYMBOL"]
    id_to_symbol.index = id_to_symbol.index.astype(str)
    expr_annotated = expr.copy()
    expr_annotated.index = expr_annotated.index.astype(str).map(id_to_symbol)
    expr_annotated = expr_annotated[expr_annotated.index.notna() & (expr_annotated.index != "")]
    expr_annotated = expr_annotated.groupby(expr_annotated.index).mean()
    gm.log.info("Annotated %d probes -> %d unique genes", expr.shape[0], expr_annotated.shape[0])
    expr_annotated.to_csv(OUTDIR / "expr_matrix_annotated.csv")

    # --- 3. Derive clean group/region labels from the "title" field -------
    sample_metadata["group"] = sample_metadata["title"].str.extract(r"_(T2D|T1D|Control)_")
    sample_metadata["region"] = sample_metadata["title"].str.extract(r"^([A-Za-z]+)_")

    mask = (sample_metadata["region"] == REGION) & (
        sample_metadata["group"].isin([CONTROL_GROUP, TREATMENT_GROUP])
    )
    meta_sub = sample_metadata.loc[mask]
    expr_sub = expr_annotated.loc[:, meta_sub.index]
    gm.log.info("Subset: %s, %s vs %s -> %d samples (%s)",
                REGION, CONTROL_GROUP, TREATMENT_GROUP, len(meta_sub),
                meta_sub["group"].value_counts().to_dict())

    # --- 4. Differential expression + insulin-gene highlighting -----------
    de = gm.run_differential_expression(expr_sub, meta_sub, "group", CONTROL_GROUP, TREATMENT_GROUP, data_kind)
    insulin_subset = gm.highlight_insulin_signaling_genes(de, species=SPECIES, mock=False)
    de.to_csv(OUTDIR / "de_results.csv")
    insulin_subset.to_csv(OUTDIR / "insulin_signaling_genes_de.csv")

    n_fdr_sig = int((de["padj"] < 0.05).sum())
    n_insulin_fdr_sig = int((insulin_subset["padj"] < 0.05).sum())
    n_insulin_raw_sig = int((insulin_subset["pvalue"] < 0.05).sum())

    # --- 5. Visualizations --------------------------------------------------
    gm.plot_volcano(de, OUTDIR / "volcano_plot.png")

    if not insulin_subset.empty:
        sig_insulin = insulin_subset.sort_values("pvalue").head(20)
        heatmap_data = np.log2(expr_sub.loc[sig_insulin.index] + 1)
        heatmap_data = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, yticklabels=True,
                    xticklabels=meta_sub["title"], cbar_kws={"label": "log2(expr) - row mean"}, ax=ax)
        ax.set_title(f"Top 20 insulin-signaling genes (by raw p) — {REGION}, "
                      f"{CONTROL_GROUP} vs {TREATMENT_GROUP} ({GSE_ID})")
        plt.tight_layout()
        fig.savefig(OUTDIR / "insulin_genes_heatmap.png", dpi=200)
        plt.close(fig)

        top_genes = sig_insulin.index[:6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, gene in zip(axes.flat, top_genes):
            plot_df = pd.DataFrame({"expression": expr_sub.loc[gene], "group": meta_sub["group"]})
            sns.boxplot(data=plot_df, x="group", y="expression", ax=ax, palette=["#1b9e77", "#d95f02"])
            sns.stripplot(data=plot_df, x="group", y="expression", ax=ax, color="black", size=5)
            ax.set_title(f"{gene} (raw p={de.loc[gene,'pvalue']:.3g}, padj={de.loc[gene,'padj']:.3g})")
        fig.suptitle(f"Top insulin-signaling genes by raw p-value (n={mask.sum()//2}/group)", y=1.02)
        plt.tight_layout()
        fig.savefig(OUTDIR / "top_insulin_genes_boxplots.png", dpi=200)
        plt.close(fig)

        top15 = insulin_subset.sort_values("pvalue").head(15).sort_values("log2FoldChange")
        colors = ["#d95f02" if p < 0.05 else "#999999" for p in top15["pvalue"]]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top15.index, top15["log2FoldChange"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"log2 fold change ({TREATMENT_GROUP} vs {CONTROL_GROUP})")
        ax.set_title("Insulin-signaling genes ranked by raw p-value\n(orange = raw p<0.05, grey = ns)")
        plt.tight_layout()
        fig.savefig(OUTDIR / "insulin_genes_forestplot.png", dpi=200)
        plt.close(fig)
    else:
        gm.log.warning("No insulin-signaling genes matched in this dataset; skipping insulin-specific plots.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(de["pvalue"].dropna(), bins=40, color="#7570b3", edgecolor="white")
    ax.set_xlabel("Raw p-value")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"P-value distribution, all genes ({REGION}, {CONTROL_GROUP} vs {TREATMENT_GROUP})")
    plt.tight_layout()
    fig.savefig(OUTDIR / "pvalue_histogram.png", dpi=200)
    plt.close(fig)

    log_expr_all = np.log2(expr_sub + 1)
    corr = log_expr_all.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = meta_sub["group"] + "_" + meta_sub.index.astype(str).str[-1]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Sample-sample correlation ({REGION})")
    plt.tight_layout()
    fig.savefig(OUTDIR / "sample_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    log_expr_t = log_expr_all.T
    log_expr_t = log_expr_t.loc[:, log_expr_t.var() > 0]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(log_expr_t - log_expr_t.mean(axis=0))
    fig, ax = plt.subplots(figsize=(6, 5))
    for group, color in [(CONTROL_GROUP, "#1b9e77"), (TREATMENT_GROUP, "#d95f02")]:
        m = (meta_sub["group"] == group).values
        ax.scatter(coords[m, 0], coords[m, 1], label=group, color=color, s=80)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA — {REGION} samples, {GSE_ID}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTDIR / "pca_plot.png", dpi=200)
    plt.close(fig)

    # --- 6. Enrichment (best-effort; requires internet, may rate-limit) ---
    enrichment_note = "not attempted"
    try:
        enrichment_real = gm.run_pathway_enrichment(de, species="mouse", outdir=OUTDIR)
        if not enrichment_real.empty:
            gm.plot_enrichment_barplot(enrichment_real, OUTDIR / "enrichment_barplot.png")
            enrichment_note = f"{len(enrichment_real)} terms returned, see enrichment_results.csv"
        else:
            enrichment_note = "Enrichr returned 0 terms"
    except Exception as exc:  # noqa: BLE001
        enrichment_note = f"failed ({exc})"
    gm.log.info("Enrichment status: %s", enrichment_note)

    # --- 7. Plain-text summary for copy-paste into a report ---------------
    summary_lines = [
        f"GSE34451 analysis summary — {REGION}, {CONTROL_GROUP} vs {TREATMENT_GROUP}",
        f"Samples: {len(meta_sub)} total ({meta_sub['group'].value_counts().to_dict()})",
        f"Platform: {GPL_ID} (Agilent rat whole-genome microarray)",
        f"Genes tested (after probe->symbol annotation, multi-probe collapse): {len(de)}",
        f"Genes FDR-significant (padj<0.05), all genes: {n_fdr_sig}",
        f"Insulin-signaling-pathway genes found in dataset: {len(insulin_subset)}",
        f"  - FDR-significant (padj<0.05): {n_insulin_fdr_sig}",
        f"  - Nominally significant (raw p<0.05, uncorrected): {n_insulin_raw_sig}",
        f"Enrichment (Enrichr, mouse-ortholog library, approximation for rat): {enrichment_note}",
        "",
        "Top 10 insulin-signaling genes by raw p-value:",
        insulin_subset.sort_values("pvalue").head(10)[["log2FoldChange", "pvalue", "padj"]].to_string(),
        "",
        "LIMITATION: n=3 per group per region (the real sample size of this public dataset). "
        "Treat all findings here as exploratory/hypothesis-generating, not confirmatory.",
    ]
    summary_text = "\n".join(summary_lines)
    (OUTDIR / "summary.txt").write_text(summary_text)
    print(summary_text)
    gm.log.info("All outputs written to %s", OUTDIR.resolve())


if __name__ == "__main__":
    main()
