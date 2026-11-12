# FLEX Ensemble Results — XGBoost + CatBoost

Run: 2026-11-11/12 · Script: `train_ensemble.py` · Data: `flex_table_ix.csv`
(276 tests; completion 28 / disruption 62 / extinction 186)
Protocol: Optuna (50 trials/model) under 5-fold Stratified CV → out-of-fold
(OOF) probabilities → weighted soft voting (weight grid-searched on OOF
Macro-F1) → per-class decision multipliers tuned on OOF Macro-F1.
Cleaning per spec: dashes → NaN; FLEX-114 pressure 0 → NaN → imputed with
the median pressure of its fuel + (O2, CO2) composition group (535 mmHg
neighbourhood); '~' stripped from FLEX-227/229 burn times (6.0 / 9.0 s);
`fuel` binarised; extinction-diameter missingness kept as NaN for
tree-native handling plus a missingness flag (Variant A).

## Variant A — "full" features (as specified, 11 features)

Features: fuel, pressure, O2, N2, CO2, He, initial diameter,
visible flame extinction diameter + missing flag, burning rate, burn time.

| Model | OOF Accuracy | OOF Macro-F1 |
|---|---|---|
| XGBoost | 0.844 | 0.698 |
| CatBoost | 0.848 | 0.699 |
| Ensemble (argmax) | 0.848 | 0.701 |
| **Ensemble (threshold-tuned)** | **0.851** | **0.705** |

Ensemble weights: **XGBoost 0.96 / CatBoost 0.04**
Class multipliers (threshold tuning): completion 1.00, disruption 1.00,
extinction 1.15

Confusion matrix (threshold-tuned; rows = true, cols = predicted,
order: completion, disruption, extinction):

```
              pred completion  pred disruption  pred extinction
true completion           8               12                8
true disruption           0               48               14
true extinction           0                7              179
```

Classification report (threshold-tuned):

```
              precision  recall  f1-score  support
completion        1.000   0.286     0.444       28
disruption        0.716   0.774     0.744       62
extinction        0.891   0.962     0.925      186
accuracy                              0.851      276
macro avg         0.869   0.674     0.705      276
```

Best hyperparameters (full precision in `ensemble_config.json`):

- XGBoost: n_estimators 400, learning_rate 0.036, max_depth 7,
  subsample 0.739, colsample_bytree 0.560, reg_alpha 0.077,
  reg_lambda 0.075, min_child_weight 8.92, gamma 0.797,
  class weights: none (Optuna preferred no weighting)
- CatBoost: iterations 550, learning_rate 0.192, depth 3,
  l2_leaf_reg 25.18, random_strength 3.49, bagging_temperature 0.70,
  auto_class_weights: None

## Variant B — "pre-ignition" features only (leakage-free, 7 features)

Only quantities known before ignition: fuel, pressure, O2, N2, CO2, He,
initial diameter. This variant can honestly predict a *future* test.

| Model | OOF Accuracy | OOF Macro-F1 |
|---|---|---|
| XGBoost | 0.797 | 0.652 |
| CatBoost | 0.801 | 0.631 |
| Ensemble (argmax) | 0.812 | 0.667 |
| **Ensemble (threshold-tuned)** | **0.819** | **0.693** |

Ensemble weights: XGBoost 0.74 / CatBoost 0.26; multipliers:
completion 1.00, disruption 0.50, extinction 1.00.

## Reading the results (honest notes)

- The gap between A and B is small (0.705 vs 0.693 tuned Macro-F1), so most
  of the signal genuinely comes from pre-ignition conditions — the
  during/after-burn features in A add little despite their leakage
  potential. For the challenge submission, B is the defensible headline
  model; A is the as-specified run.
- Weak spot: **completion recall is only 0.29–0.43**. Most true completions
  are predicted as disruption/extinction — with 28 examples the class is
  intrinsically hard, and no class-weighting scheme Optuna tried fixed it.
- Threshold tuning was performed on the same OOF probabilities it is
  evaluated on, so the tuned Macro-F1 is mildly optimistic; the argmax
  numbers are the unbiased comparison.
- The Variant A ensemble weight (0.96 XGBoost) means that ensemble is
  effectively XGBoost with a CatBoost nudge; in Variant B the blend is
  more balanced (0.74 / 0.26).
- Majority-class reference: always predicting "extinction" gives 0.674
  accuracy but only ~0.27 Macro-F1.

## Files

- `train_ensemble.py` — full pipeline (cleaning, HPT, OOF, ensembling)
- `ensemble_config.json` — all metrics, confusion matrices, best params
- `oof_predictions.csv` — per-test OOF probabilities and predictions (A)
- `model_xgboost_A_full.json`, `model_catboost_A_full.cbm`,
  `model_xgboost_B_preignition.json`, `model_catboost_B_preignition.cbm`
  — final models refit on all 276 rows with the best hyperparameters
