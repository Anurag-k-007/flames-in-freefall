#!/usr/bin/env python3
"""
Flame in Freefall — FLEX extinction-outcome dashboard (NASA Space Apps 2026).

Headline model: Variant B (leakage-free, pre-ignition features only),
XGBoost + CatBoost weighted soft-voting ensemble, OOF accuracy 0.819,
macro-F1 0.693 (threshold-tuned; argmax 0.812 / 0.667).
Variant A (all 11 features, incl. during/after-burn) is shown as the
as-specified comparison only — it cannot honestly predict a future test.

Run:  pip install -r requirements.txt
      streamlit run app.py
Data: flex_table_ix.csv — NASA/TP-2015-216046, Table IX (276 FLEX tests).
Models: model_*_B_preignition.* / model_*_A_full.* in this folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).parent
CLASSES = ["completion", "disruption", "extinction"]
# From ensemble_config.json (Optuna 50 trials/model, 5-fold stratified CV)
CONFIG = {
    "B_preignition": {
        "label": "Variant B — pre-ignition only (headline)",
        "features": ["fuel_bin", "pressure_mmhg", "o2_mole_fraction",
                     "n2_mole_fraction", "co2_mole_fraction",
                     "he_mole_fraction", "initial_diameter_mm"],
        "w_xgb": 0.74, "w_cat": 0.26,
        "multipliers": np.array([1.0, 0.5, 1.0]),  # completion, disruption, extinction
        "xgb": "model_xgboost_B_preignition.json",
        "cat": "model_catboost_B_preignition.json",
        "acc": 0.819, "f1": 0.693, "acc_argmax": 0.812, "f1_argmax": 0.667,
    },
    "A_full": {
        "label": "Variant A — all features (comparison only)",
        "features": ["fuel_bin", "pressure_mmhg", "o2_mole_fraction",
                     "n2_mole_fraction", "co2_mole_fraction",
                     "he_mole_fraction", "initial_diameter_mm",
                     "visible_flame_extinction_diameter_mm",
                     "extinction_diameter_missing", "burning_rate_mm2_s",
                     "burn_time_s"],
        "w_xgb": 0.96, "w_cat": 0.04,
        "multipliers": np.array([1.0, 1.0, 1.15]),
        "xgb": "model_xgboost_A_full.json",
        "cat": "model_catboost_A_full.json",
        "acc": 0.851, "f1": 0.705, "acc_argmax": 0.848, "f1_argmax": 0.701,
    },
}


@st.cache_resource
def load_models(variant):
    cfg = CONFIG[variant]
    xgb = XGBClassifier()
    xgb.load_model(str(BASE / cfg["xgb"]))
    cat = CatBoostClassifier()
    cat.load_model(str(BASE / cfg["cat"]), format="json")
    return xgb, cat


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "flex_table_ix.csv")


def ensemble_probs(variant, row):
    """row: dict of feature name -> value. Returns (probs, multipliers-applied scores)."""
    cfg = CONFIG[variant]
    xgb, cat = load_models(variant)
    X = np.array([[row[f] for f in cfg["features"]]], dtype=float)
    probs = cfg["w_xgb"] * xgb.predict_proba(X)[0] + cfg["w_cat"] * cat.predict_proba(X)[0]
    return probs, probs * cfg["multipliers"]


st.set_page_config(page_title="Flame in Freefall — FLEX predictor", layout="wide")
st.title("🔥 Flame in Freefall — will this droplet flame go out?")
st.markdown(
    "Microgravity droplet combustion from NASA's **FLEX** experiment on the ISS. "
    "Pick the conditions **before ignition** and the trained ensemble predicts the outcome: "
    "**extinction** (flame goes out), **disruption** (droplet breaks up) or **completion** "
    "(fuel burns fully). Data: NASA/TP-2015-216046, Table IX — 276 tests."
)

df = load_data()

# ---------------------------------------------------------------- predictor
st.header("Predict a new test")
col1, col2 = st.columns(2)
with col1:
    fuel = st.selectbox("Fuel", ["methanol", "heptane"])
    pressure = st.slider("Pressure (mmHg)", 526, 2320, 760, step=10)
    o2 = st.slider("O₂ mole fraction", 0.12, 0.34, 0.21, step=0.01)
    diameter = st.slider("Initial droplet diameter (mm)", 1.1, 4.67, 3.0, step=0.05)
with col2:
    co2 = st.slider("CO₂ mole fraction", 0.0, 0.70, 0.0, step=0.01)
    he = st.slider("He mole fraction", 0.0, 0.45, 0.0, step=0.01)
    n2 = 1.0 - o2 - co2 - he
    st.metric("N₂ (balance, auto-computed)", f"{n2:.2f}")
    if n2 < 0:
        st.error("O₂ + CO₂ + He exceeds 1.0 — lower one of them; N₂ cannot be negative.")
    variant = st.radio("Model", ["B_preignition", "A_full"],
                       format_func=lambda v: CONFIG[v]["label"])

if n2 >= 0:
    row_b = {"fuel_bin": 1 if fuel == "heptane" else 0, "pressure_mmhg": pressure,
             "o2_mole_fraction": o2, "n2_mole_fraction": n2,
             "co2_mole_fraction": co2, "he_mole_fraction": he,
             "initial_diameter_mm": diameter}
    probs, scores = ensemble_probs(variant, row_b)
    pred = CLASSES[int(np.argmax(scores))]
    st.subheader(f"Predicted outcome: **{pred}**")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES))
    st.caption(
        "Probabilities are the weighted ensemble output (XGBoost "
        f"{CONFIG[variant]['w_xgb']:.2f} / CatBoost {CONFIG[variant]['w_cat']:.2f}); "
        "the decision applies the per-class multipliers tuned for macro-F1. "
        f"OOF accuracy {CONFIG[variant]['acc']:.3f}, macro-F1 {CONFIG[variant]['f1']:.3f} "
        f"(argmax, unbiased: {CONFIG[variant]['acc_argmax']:.3f} / {CONFIG[variant]['f1_argmax']:.3f})."
    )
    if variant == "A_full":
        st.warning(
            "Variant A uses burning rate, burn time and extinction diameter — only known "
            "during/after a burn — so it cannot honestly predict a future test. "
            "Headline Variant B for the submission."
        )

# ---------------------------------------------------------------- explorer
st.header("Explore the 276 FLEX tests")
st.markdown("Outcomes in the data: extinction 186 · disruption 62 · completion 28.")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Outcome by fuel")
    st.bar_chart(df.groupby(["fuel", "test_end"]).size().unstack(fill_value=0))
with c2:
    st.subheader("Outcome by O₂ band")
    tmp = df.copy()
    tmp["o2_band"] = pd.cut(tmp["o2_mole_fraction"], [0.1, 0.18, 0.24, 0.35],
                            labels=["0.12–0.18", "0.18–0.24", "0.24–0.34"])
    st.bar_chart(tmp.groupby(["o2_band", "test_end"], observed=True).size().unstack(fill_value=0))

fuel_f = st.multiselect("Filter by fuel", ["methanol", "heptane"], default=["methanol", "heptane"])
out_f = st.multiselect("Filter by outcome", CLASSES, default=CLASSES)
st.dataframe(df[df["fuel"].isin(fuel_f) & df["test_end"].isin(out_f)], use_container_width=True)
st.download_button("Download the FLEX CSV", df.to_csv(index=False),
                   file_name="flex_table_ix.csv", mime="text/csv")

# ---------------------------------------------------------------- honesty box
st.header("What this model can and cannot say")
st.markdown(
    "- **Weak spot: completion recall is only 0.29–0.43** — with just 28 completion examples, "
    "most true completions are predicted as disruption/extinction. Do not present completion "
    "predictions as reliable.\n"
    "- Threshold-tuned macro-F1 is mildly optimistic (tuning and evaluation share the same "
    "out-of-fold predictions); the argmax numbers are the unbiased comparison.\n"
    "- FLEX-114's pressure is printed as 0 in the report (a misprint); it was treated as "
    "missing. FLEX-227/229 burn times are approximate (~6 s / ~9 s).\n"
    "- Official PSI resources for this challenge were due **Fri 13 Nov 2026** — swap/extend "
    "this dataset when they land; this FLEX table is the trainable stand-in built 11 Nov."
)
