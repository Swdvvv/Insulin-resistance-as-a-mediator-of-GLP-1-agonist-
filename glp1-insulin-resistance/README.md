# GLP-1 / Insulin Resistance Workspace

Workspace for the research project: **Insulin resistance as a mediator of
GLP-1 receptor agonist effects**.

## Structure

```
glp1-insulin-resistance/
├── papers/         Source PDFs / references for included studies
├── datasets/        Raw or processed datasets used in the analysis
├── notes/           Search strategies, meeting notes, literature notes
├── extraction/       Data extraction worksheets per study
├── analysis/        Scripts and outputs for statistical/mediation analysis
├── figures/          Generated plots and diagrams
├── manuscript/        Drafts of the write-up (sections, full manuscript)
│
├── research_question.md   PICO framework and research question definition
├── evidence_table.csv      Master evidence table summarizing included studies
└── README.md               This file
```

## Workflow
1. Define the research question and PICO (`research_question.md`).
2. Search and store source papers (`papers/`), logging search strategy in `notes/`.
3. Screen and extract data per study into `extraction/`, summarizing in `evidence_table.csv`.
4. Store any raw/derived datasets in `datasets/`.
5. Run analysis (e.g., mediation analysis, meta-analysis) in `analysis/`, saving figures to `figures/`.
6. Draft the manuscript in `manuscript/`.

## Status
Project setup — folders initialized, evidence table and research question
scaffolded. Next step: populate `research_question.md` PICO details and begin
literature search.
