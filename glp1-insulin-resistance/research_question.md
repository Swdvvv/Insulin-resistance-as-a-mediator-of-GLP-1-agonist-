# Research Question

*Last updated: 2026-06-22. This is a living research note — update the
Evidence synthesis, Hypotheses, and Status sections as new analyses complete.*

## Working title
Central insulin resistance as a mediator of GLP-1 receptor agonist
neuroprotection in Alzheimer's and Parkinson's disease

## Primary research question
What evidence supports central insulin resistance as a mediator of GLP-1
receptor agonist neuroprotection in Alzheimer's and Parkinson's disease?

## Background / rationale
- Alzheimer's disease (AD) and Parkinson's disease (PD) have both been
  linked to impaired brain (central) insulin signaling — sometimes termed
  "type 3 diabetes" in the AD literature — contributing to neuroinflammation,
  mitochondrial dysfunction, protein aggregation (amyloid-β, tau, α-synuclein),
  and synaptic loss.
- GLP-1 receptors are expressed in the CNS (hippocampus, substantia nigra,
  cortex). GLP-1 receptor agonists (GLP-1RAs; e.g., exenatide, liraglutide,
  semaglutide) cross or signal across the blood-brain barrier and have shown
  neuroprotective effects in preclinical AD/PD models and early clinical
  trials.
- Proposed mechanistic pathway: GLP-1RAs restore central insulin
  sensitivity/signaling (PI3K-Akt pathway, GSK-3β inhibition, reduced
  neuroinflammation) → downstream reduction in pathological protein
  aggregation and neuronal loss → clinical neuroprotection (cognitive/motor
  outcomes).
- This project evaluates the evidence base for central insulin resistance
  as the mediating mechanism, as opposed to GLP-1RA effects on neuroprotection
  occurring independently of insulin signaling (e.g., via direct
  anti-inflammatory or anti-apoptotic effects).

## Working hypotheses

Three candidate hypotheses were formulated and weighed against the evidence
synthesized below (see `extraction/contradicting_evidence.json` for the full
reasoning). They are listed here in decreasing order of how well each is
*currently* supported — none are confirmed; this ranking reflects which is
*least contradicted* by the evidence gathered so far.

| # | Hypothesis | Current assessment |
|---|---|---|
| **H3** | Brain insulin resistance represents a shared pathological mechanism linking AD and PD and **contributes to (not necessarily "primarily explains")** the therapeutic effects of GLP-1RAs. | **Best-supported framing.** Appropriately scoped — doesn't overclaim primacy, matches the cross-disease structure of the evidence (insulin resistance independently documented in both AD and PD), and isn't directly contradicted by any single finding. Not yet *confirmed* — no formal mediation test exists (see Evidence gaps). |
| **H1** | GLP-1RAs exert neuroprotective effects **primarily** through restoration of PI3K-AKT signaling impaired by central insulin resistance. | **Contradicted by decoupling evidence.** Multiple studies show the predicted mechanism and the neuroprotective phenotype moving independently of each other (e.g., Carranza-Naval et al. 2021: full neuroprotective phenotype with *zero* change in insulin-receptor-pathway mRNA; Paladugu et al. 2021: pAkt unchanged despite other improvements). The "primarily" claim is too strong given multi-mechanism evidence (anti-inflammatory, autophagy, mitochondrial effects also implicated). |
| **H2** | Metabolic dysfunction (diabetes/obesity/insulin resistance) identifies a **responder subgroup** for GLP-1 therapy in neurodegenerative disease. | **Weakest fit.** The one direct empirical test (Athauda et al. 2019 post-hoc analysis) found BMI and insulin resistance did *not* predict treatment response — phenotype and disease stage did. Most RCTs excluded diabetics by design, so this hypothesis is largely untestable with current trial data; the one trial with a real diabetic subgroup (evoke/evoke+, ~14%) has not published a stratified analysis. |

## PICO framework
- **Population:** Adults with Alzheimer's disease, mild cognitive impairment,
  Parkinson's disease, or relevant preclinical models (animal/cell) of either
- **Intervention:** GLP-1 receptor agonist therapy (exenatide, liraglutide,
  semaglutide, lixisenatide, dulaglutide, dual/triple agonists, etc.)
- **Comparator:** Placebo, standard care, or untreated controls
- **Outcome:** Neuroprotection — cognitive decline/function (AD), motor
  function/progression (PD), neuroimaging biomarkers, pathological markers
  (amyloid-β, tau, α-synuclein, neuroinflammation)
  **Mediator:** Central/brain insulin resistance (CSF insulin signaling
  markers, brain PI3K-Akt/IRS-1 activity, neuroimaging-based insulin
  sensitivity measures, surrogate peripheral markers where central measures
  are unavailable)

## Evidence synthesis to date

Full structured extractions live in `extraction/` (see References below); this
section summarizes the headline findings for manuscript-writing purposes.

### What's reasonably well supported
- **Central insulin resistance co-occurs with AD and PD pathology**,
  independently documented in both diseases: ex vivo human AD hippocampal
  tissue shows 29–90% reduced insulin/IRS-1 activation (Talbot 2014); PD brain
  insulin signaling is described as "desensitised" across multiple sources
  (Mulvaney et al. 2020 Cochrane review and others); GEO transcriptomic
  datasets (GSE262426, GSE306976, GSE34451) independently show
  insulin-resistance-associated gene expression changes in AD models, though
  **none of these GEO datasets include a GLP-1RA treatment arm**.
- **GLP-1R and insulin-receptor signaling converge on shared downstream
  nodes** (PI3K/Akt, GSK-3β, mTOR, cAMP/PKA) — established mechanistically,
  though described in the literature as *parallel/convergent* pathways, not
  direct receptor-receptor crosstalk.
- **Some preclinical GLP-1RA studies show parallel changes** in both an
  insulin-signaling marker and a neurodegenerative outcome in the same
  animals (e.g., Zhang et al. 2022, 6-OHDA PD rat model: pIRS-1(Ser312)
  normalization paralleled improvements in alpha-synuclein, dopaminergic
  neuron counts, and behavior).
- **One human clinical biomarker dataset exists**: Athauda et al. 2019
  (secondary analysis of the Exenatide-PD RCT) measured neuronal-exosome
  IRS-1/Akt/mTOR changes in actual PD patients and found these associated
  with motor outcome change — the single strongest piece of human in-vivo
  evidence, though it stops short of formal mediation testing.

### What actively cuts against a simple causal story
See `extraction/contradicting_evidence.json` for full detail and citations.
In brief: several studies show neuroprotection occurring *without* the
predicted insulin-pathway marker changing (decoupling), the literal
proposed clinical marker (cerebral glucose metabolism) has been null in
more than one AD trial even when other benefits appeared, pooled
meta-analytic clinical evidence is mostly null/very-low-certainty for both
diseases, and large observational effect sizes (e.g., HR 0.30 for dementia
incidence) are discordant with null RCTs in the same disease area — a
classic confounding signature, not confirmatory evidence.

### Metabolic-subgroup question ("do metabolically impaired patients benefit
more?")
See `extraction/metabolic_subgroup_extraction.json`. Largely unanswerable
with current data: 5 of 7 identified RCTs excluded diabetics by design
specifically to isolate a CNS effect from peripheral glycemic confounding.
The one trial with a meaningful diabetic subgroup (evoke/evoke+, ~14%,
n≈516) has not published a diabetes-stratified result; an early WebSearch
summary falsely claimed such a subgroup analysis existed and "supported the
main results" — this was traced to a hallucinated/unsupported inference by
a search-summarization tool, not a real finding, and is documented as a
caught correction in that file's `corrections_log`.

## Methodology: analytical tools developed

Because no paper in this literature has ever run a formal statistical test
of the mediation hypothesis, this project built the missing analytical
infrastructure rather than relying solely on narrative synthesis:

- **`analysis/glp1_model.py`** — a unified model in three stages:
  1. *GEO transcriptomics pipeline* (`geo` subcommand): download → differential
     expression → pathway enrichment → insulin-signaling-gene highlighting,
     runnable on real GEO series (tested successfully end-to-end on GSE34451)
     or in a fully offline `--dry-run` mode.
  2. *Causal mediation model* (`mediation` subcommand): a Baron-Kenny path
     model with bootstrapped indirect-effect confidence intervals
     (`fit_mediation`), ready to run on real individual-patient data the
     moment it exists; a calibrated-but-clearly-labeled illustration based on
     Athauda et al. 2017/2019's real aggregate statistics
     (`--simulate-illustration`); and a power/feasibility analysis
     (`--power-analysis`) testing whether trials of the sizes actually used in
     this literature (n=62, n=204, n=3,808) were even capable of detecting
     mediation if it exists.
  3. *Integrated pipeline demo* (`pipeline` subcommand): demonstrates how
     Stage 1's output would feed Stage 2's input once a real linked dataset
     exists (it doesn't yet) — illustrative plumbing, not a finding.
- **`analysis/adni_pipeline.py`** — an observational secondary-data-analysis
  pipeline against the ADNI cohort: flags GLP-1RA medication use from the
  medication log, computes HOMA-IR (a *peripheral*, not central, insulin
  resistance proxy) from glucose/insulin labs, and runs the same mediation
  model against cognitive/MRI decline. Requires the researcher's own approved
  ADNI data access (gated dataset; not downloadable by this tooling). Has a
  `--demo` mode using synthetic schema-matched data to verify the pipeline
  runs before real access is obtained.
- Both are unit-tested offline (`analysis/tests/`, 30+ tests total) and were
  verified to actually execute correctly in a real Python 3.12 environment
  (Google Colab), not just code-reviewed.

## Secondary questions
1. What direct evidence exists (vs. inferred/correlational) that GLP-1RAs
   improve central insulin signaling in AD/PD models or patients?
   → **Partially answered**: direct preclinical marker evidence exists in
   several models (see Evidence synthesis); the only human biomarker data is
   Athauda et al. 2019 (PD only, association not mediation).
2. Do preclinical mechanistic studies (cell/animal) support a causal chain
   from GLP-1RA → restored central insulin signaling → reduced pathology?
   → **Mixed**: some studies show the full chain moving together; others
   show clear decoupling (see `contradicting_evidence.json`).
3. Do clinical trials reporting cognitive/motor outcomes also measure or
   infer changes in central insulin resistance, and is improvement in one
   associated with improvement in the other?
   → **Mostly no**: most RCTs do not measure a central insulin-resistance
   marker at all; where a proxy outcome exists (cerebral glucose metabolism),
   it has frequently been null.
4. Are effects consistent across AD and PD, or does the mediating pathway
   differ by disease/protein aggregate type?
   → **Inconsistent**: PD trials more often hit significance on primary motor
   endpoints; AD trials (including the largest, evoke/evoke+) have been
   predominantly null on primary cognitive endpoints. Disease type and
   metabolic-exclusion criteria are confounded in the current evidence base,
   so this difference cannot yet be attributed to the mediating pathway itself.
5. What is the relative strength of the insulin-resistance-mediation
   hypothesis versus competing/parallel mechanisms (anti-inflammatory,
   anti-apoptotic, mitochondrial, GLP-1R-direct neurotrophic effects)?
   → **Likely contributory, not dominant**: reviews and the contradicting-
   evidence catalog both point toward a multi-mechanism explanation in which
   insulin-pathway restoration is one contributor among several, not proven
   to be necessary or sufficient on its own.

## Inclusion / exclusion criteria (draft)
- **Include:** RCTs, cohort studies, and mechanistic preclinical (animal/cell)
  studies evaluating GLP-1RA treatment with reported central insulin
  signaling/resistance measures and a neurodegenerative (AD or PD)
  outcome or pathology marker.
- **Exclude:** Studies of GLP-1RAs limited to peripheral metabolic outcomes
  with no neurological/neurodegenerative endpoint or mechanism discussed;
  case reports without mechanistic or outcome data.

## Known evidence gaps (for the Discussion/Limitations section)
1. **No formal statistical mediation analysis exists** in any paper reviewed
   (21+ papers spanning RCTs, preclinical studies, meta-analyses, and a
   Cochrane review) — confirmed exhaustively; this is the central gap this
   project's `mediation_model` component was built to eventually fill.
2. **No public dataset links GLP-1RA treatment, a central insulin-resistance
   marker, and a neurodegenerative outcome in the same subjects** — confirmed
   across both the literature search and a dedicated GEO transcriptomics
   search (GSE262426/GSE306976/GSE34451 have the insulin-resistance side but
   no GLP-1RA arm; GSE41345 has a GLP-1RA arm but a TBI, not AD/PD, model).
3. **Most RCTs structurally exclude the population (diabetics) needed to test
   the metabolic-subgroup hypothesis** — by design, to isolate a CNS effect
   from peripheral glycemic confounding.
4. **KEGG's curated reference pathways for AD (hsa05010) and PD (hsa05012) do
   not include insulin-signaling components** — the hypothesized mechanism is
   not yet reflected in canonical pathway curation, only in primary/review
   literature.
5. **Observational real-world studies show effect sizes inconsistent with
   RCTs in the same disease area** (e.g., AbuAlrob et al. 2025's HR 0.30 vs.
   evoke/evoke+'s null primary result) — a likely confounding signature
   rather than confirmatory evidence, and a caution against over-weighting
   observational data in the eventual manuscript.

## References to supporting files
- `extraction/literature_extraction.json` — full structured extraction of 21
  papers (disease, drug, IR evidence, signaling markers, outcomes, mediation
  testing, conclusions) plus a `synthesis_summary` and
  `additional_database_sources` (GEO, DrugBank, KEGG, OpenAlex, Cochrane).
- `extraction/contradicting_evidence.json` — catalog of evidence cutting
  against the simple mediation hypothesis, organized by contradiction type.
- `extraction/metabolic_subgroup_extraction.json` — trial-level extraction
  addressing "do metabolically impaired patients benefit more," including a
  documented correction of a caught search-tool hallucination.
- `extraction/entities.json` / `extraction/triples.csv` — gene/protein/
  pathway/cell-type/disease-process entities and subject→relation→object
  triples mined from the literature, for knowledge-graph-style review.
- `evidence_table.csv` — flat per-study summary table.
- `analysis/glp1_model.py`, `analysis/adni_pipeline.py` — the analytical
  tools described in Methodology above, with offline pytest suites in
  `analysis/tests/`.

## Status
- [x] Finalize primary research question
- [x] Formulate and rank competing hypotheses (H1/H2/H3) against evidence
- [x] Complete literature search and structured extraction (21+ papers)
- [x] Identify and catalog contradicting evidence
- [x] Investigate metabolic-subgroup question (evidence gap identified)
- [x] Search GEO/KEGG/DrugBank/OpenAlex/Cochrane for supporting database evidence
- [x] Build mediation-analysis tooling (`glp1_model.py`, `adni_pipeline.py`)
- [x] Verify tooling actually executes (Google Colab, Python 3.12)
- [ ] Obtain real ADNI data access and run `adni_pipeline.py` in real mode
- [ ] Run `glp1_model.py geo` on a real dataset through to enrichment results
      (GSE34451 download confirmed working; contrast labels not yet finalized)
- [ ] Draft manuscript sections using this note as the Background/Discussion
      skeleton (see `manuscript/`)
