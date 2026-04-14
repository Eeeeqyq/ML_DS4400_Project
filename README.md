# DS4400 Final Project

This repo contains our milestone report, figure-generation code, and generated figures for a DS4400 project predicting **estimated college debt payoff time** from institutional features.

## Project demo

Live project demo: [https://demo-theta-orcin-89.vercel.app/](https://demo-theta-orcin-89.vercel.app/)

## What’s in this repo

- `milestone.html`: milestone report (formatted to paste cleanly into Google Docs)
- `milestone.md`: milestone report (markdown version)
- `proposal.md`: original project proposal
- `final.md`: placeholder for final report (to be written)
- `generate_figures.py`: script that reproduces all EDA + model figures and `model_results.json`
- `figures/`: the 7 PNG figures referenced in the report
- `model_results.json`: Ridge + Decision Tree metrics from the script

## Getting the dataset (not committed to GitHub)

The College Scorecard dataset is **not** committed to this repository because it’s very large.

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

Run:

```bash
conda run -n ds python generate_figures.py
```

Outputs:

- `figures/01_target_distribution.png`
- `figures/02_feature_availability.png`
- `figures/03_correlation_heatmap.png`
- `figures/04_payoff_by_control.png`
- `figures/05_tuition_vs_payoff.png`
- `figures/06_model_comparison.png`
- `figures/07_predicted_vs_actual.png`
- `model_results.json`

