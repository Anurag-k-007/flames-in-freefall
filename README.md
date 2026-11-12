# Flame in Freefall — Will This Droplet Flame Go Out?

A web tool that predicts the outcome of a microgravity droplet-combustion
test **before ignition**, built for the NASA Space Apps Challenge 2026
challenge *Flame in Freefall: AI-Powered Fire Safety Insights from
Microgravity Combustion Data*.

On the ISS there is no "up" for hot air to rise, so a droplet flame behaves
very differently from a fire on Earth — and knowing whether it will go out,
break up, or burn to completion matters for spacecraft fire safety.

## The app

`app.py` (Streamlit) provides:

- **Predictor** — pick pre-ignition conditions (fuel, pressure, O₂/CO₂/He
  mix, droplet size) with sliders and get the predicted outcome with
  probabilities from a soft-voting ensemble (XGBoost + CatBoost).
- **Explorer** — outcome charts by fuel and oxygen band, the full
  filterable 276-row FLEX table, and a CSV download.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

276 real ISS droplet-combustion tests from NASA's Flame Extinguishment
Experiment (FLEX), published as **NASA/TP-2015-216046, Table IX** (Dietrich
et al., NASA Glenn Research Center), downloaded from the NASA Technical
Reports Server (NTRS ID 20150023456). Challenge data source: NASA Physical
Sciences Informatics (psi.nasa.gov).

Each row: methanol or n-heptane droplet, ambient pressure, O₂/N₂/CO₂/He mole
fractions, initial droplet diameter, burning rate, burn time, extinction
diameter, and the recorded outcome — **extinction (186), disruption (62),
completion (28)**. Test numbers 156, 205, 213 and 228 are absent from the
report itself. Two data quirks are preserved and documented in the code:
FLEX-114's pressure is printed as 0 in the report (treated as missing and
imputed from its composition group), and the burn times of FLEX-227 (~6 s)
and FLEX-229 (~9 s) are approximate in the original.

## Model

`train_ensemble.py` — XGBoost and CatBoost, each tuned with Optuna
(50 trials) under 5-fold stratified cross-validation; out-of-fold
probabilities combined by weighted soft voting; per-class decision
multipliers tuned for macro-F1. Full metrics in `results_summary.md` and
`ensemble_config.json`.

- **Variant B (headline, pre-ignition features only):** accuracy **0.819**,
  macro-F1 **0.693** — this is the model that can honestly predict a future
  test.
- **Variant A (comparison, adds during/after-burn measurements):**
  accuracy 0.851, macro-F1 0.705.

Stated limit: completion recall is only 0.29–0.43, because just 28 of the
276 tests ended in completion. This tool is a fire-safety insight aid, not
a safety certification.

## Files

- `app.py` — the Streamlit app
- `flex_table_ix.csv` — the dataset
- `train_ensemble.py` — training pipeline
- `model_*.json` — trained XGBoost and CatBoost models (JSON exports)
- `ensemble_config.json`, `results_summary.md` — metrics and provenance

Data: NASA (public domain). Built with open-source tools; not affiliated
with or endorsed by NASA.
