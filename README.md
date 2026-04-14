# DS4400 Final Project

This repo contains our milestone report, figure-generation code, and generated figures for a DS4400 project predicting **estimated college debt payoff time** from institutional features.

## Project demo

Live project demo: [https://demo-theta-orcin-89.vercel.app/](https://demo-theta-orcin-89.vercel.app/)

## What’s in this repo

- `milestone.html`: milestone report (formatted to paste cleanly into Google Docs)
- `milestone.md`: milestone report (markdown version)
- `proposal.md`: original project proposal
- `final.md`: placeholder for final report (to be written)
- `scripts/generate_figures.py`: reproduces all EDA + model figures and JSON metric outputs
- `scripts/figures/`: PNG figures produced by that script (21 files, numbered `01_`–`21_`)
- `scripts/model_results.json`: test metrics for Ridge, Decision Tree, Random Forest, and Neural Network
- `scripts/feature_comparison.json`, `scripts/capped_diagnostic.json`: additional diagnostics from the same script
- `scripts/export_model.py`: trains the Random Forest and writes `demo/public/model.json` for the browser demo

## Getting the dataset (not committed to GitHub)

The College Scorecard dataset is **not** committed to this repository because it is very large.

1. Download the data from the U.S. Department of Education College Scorecard site:
   - `https://collegescorecard.ed.gov/data`
2. Use the “All Data Files” download and extract it.
3. Copy the extracted folder into this repo at:

   `raw-data/`

4. Confirm this file exists:

   `raw-data/Most-Recent-Cohorts-Institution.csv`

Note: the official extracted folder name is date-stamped and changes over time. As long as
`Most-Recent-Cohorts-Institution.csv` exists somewhere under `raw-data/`, the script will find it.

## Reproducing figures + results (uses conda env `ds`)

From the **repository root**:

```bash
conda run -n ds python scripts/generate_figures.py
```

This writes PNGs under `scripts/figures/` and JSON under `scripts/`.

**Figure outputs** (in order):

- `01_target_distribution.png` through `07_high_earning_score_vs_payoff.png` — EDA
- `08_ridge_alpha_sweep.png` through `11_nn_sweeps.png` — hyperparameter sweeps
- `12_model_comparison.png` through `21_residuals_by_type.png` — models, diagnostics, importances

**JSON outputs:**

- `scripts/model_results.json`
- `scripts/feature_comparison.json`
- `scripts/capped_diagnostic.json`

**Browser demo model** (after pipeline changes, keep in sync):

```bash
conda run -n ds python scripts/export_model.py
```

Writes `demo/public/model.json`.
