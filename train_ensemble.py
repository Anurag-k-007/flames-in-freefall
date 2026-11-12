#!/usr/bin/env python3
"""
Flame in Freefall - FLEX extinction outcome model.

Optimized ensemble: XGBoost + CatBoost, each tuned with Optuna under
5-fold Stratified CV, combined by weighted soft voting on out-of-fold (OOF)
probabilities, with per-class decision-multiplier tuning for Macro-F1.

Two feature variants are trained:
  A "full"        - all measured features, incl. during/after-burn ones
                    (burning_rate, burn_time, visible_flame_extinction_diameter
                    + its missingness flag). Matches the requested spec.
  B "preignition" - leakage-free: only quantities known before ignition
                    (fuel, pressure, gas composition, initial diameter).
                    This is the variant that can honestly predict a future test.

Data: flex_table_ix.csv (NASA/TP-2015-216046, Table IX).
Target: test_end in {completion, disruption, extinction}  (28 / 62 / 186)
"""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
N_SPLITS = 5
N_TRIALS = 50
rng = np.random.RandomState(SEED)

# ---------------------------------------------------------------- cleaning
def load_and_clean(path="flex_table_ix.csv"):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    num_cols = ["pressure_mmhg", "o2_mole_fraction", "n2_mole_fraction",
                "co2_mole_fraction", "he_mole_fraction", "initial_diameter_mm",
                "visible_flame_extinction_diameter_mm", "burning_rate_mm2_s",
                "burn_time_s"]
    for c in num_cols:  # dashes -> NaN, strip '~' approximations -> numeric
        df[c] = (df[c].str.replace("~", "", regex=False)
                      .replace({"–": np.nan, "-": np.nan, "": np.nan})
                      .astype(float))
    # FLEX-114: pressure printed as 0 in the report -> missing.
    df.loc[df["pressure_mmhg"] == 0, "pressure_mmhg"] = np.nan
    # Impute pressure by median of same fuel + same (O2, CO2) composition
    # group (FLEX-114's neighbours at that composition sit at ~535 mmHg).
    grp = df.groupby(["fuel", "o2_mole_fraction", "co2_mole_fraction"])[
        "pressure_mmhg"].transform("median")
    df["pressure_mmhg"] = df["pressure_mmhg"].fillna(grp)
    df["pressure_mmhg"] = df["pressure_mmhg"].fillna(
        df.groupby("fuel")["pressure_mmhg"].transform("median"))
    df["pressure_mmhg"] = df["pressure_mmhg"].fillna(df["pressure_mmhg"].median())
    # Categorical: fuel -> binary. Missingness of extinction diameter kept as
    # NaN for tree-native handling + explicit indicator flag.
    df["fuel_bin"] = (df["fuel"].str.lower() == "heptane").astype(int)
    df["extinction_diameter_missing"] = df[
        "visible_flame_extinction_diameter_mm"].isna().astype(int)
    return df

FEATURES_A = ["fuel_bin", "pressure_mmhg", "o2_mole_fraction",
              "n2_mole_fraction", "co2_mole_fraction", "he_mole_fraction",
              "initial_diameter_mm", "visible_flame_extinction_diameter_mm",
              "extinction_diameter_missing", "burning_rate_mm2_s",
              "burn_time_s"]
FEATURES_B = ["fuel_bin", "pressure_mmhg", "o2_mole_fraction",
              "n2_mole_fraction", "co2_mole_fraction", "he_mole_fraction",
              "initial_diameter_mm"]

# ---------------------------------------------------------------- models
def sample_weights(y_tr, mode):
    if mode == "none":
        return None
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    if mode == "sqrt":
        cw = np.sqrt(cw)
    m = dict(zip(classes, cw))
    return np.array([m[v] for v in y_tr])

def make_xgb(p, seed=SEED):
    return XGBClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"], min_child_weight=p["min_child_weight"],
        gamma=p["gamma"], objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=seed, n_jobs=2,
        tree_method="hist")

def make_cat(p, seed=SEED):
    return CatBoostClassifier(
        iterations=p["iterations"], learning_rate=p["learning_rate"],
        depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"],
        random_strength=p["random_strength"],
        bagging_temperature=p["bagging_temperature"],
        auto_class_weights=p["auto_class_weights"], nan_mode="Min",
        loss_function="MultiClass", random_seed=seed, verbose=0,
        thread_count=2)

def cv_macro_f1(make, params, cw_mode, X, y, skf):
    """Mean 5-fold macro-F1 (plain argmax) for one hyperparameter set."""
    scores = []
    for tr, te in skf.split(X, y):
        m = make(params)
        if isinstance(m, XGBClassifier):
            m.fit(X[tr], y[tr], sample_weight=sample_weights(y[tr], cw_mode))
        else:
            m.fit(X[tr], y[tr])
        scores.append(f1_score(y[te], m.predict(X[te]), average="macro"))
    return float(np.mean(scores))

def tune_xgb(X, y, skf):
    def obj(t):
        p = dict(
            n_estimators=t.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=t.suggest_int("max_depth", 2, 8),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            min_child_weight=t.suggest_float("min_child_weight", 0.5, 10.0),
            gamma=t.suggest_float("gamma", 0.0, 5.0),
            cw_mode=t.suggest_categorical("cw_mode", ["none", "balanced", "sqrt"]))
        return cv_macro_f1(make_xgb, p, p["cw_mode"], X, y, skf)
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return s.best_params, s.best_value

def tune_cat(X, y, skf):
    def obj(t):
        p = dict(
            iterations=t.suggest_int("iterations", 100, 800, step=50),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=t.suggest_int("depth", 2, 8),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0),
            random_strength=t.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 5.0),
            auto_class_weights=t.suggest_categorical(
                "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]))
        return cv_macro_f1(make_cat, p, "none", X, y, skf)
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return s.best_params, s.best_value

# ---------------------------------------------------------------- ensemble
def oof_probs(make, params, cw_mode, X, y, skf):
    oof = np.zeros((len(y), 3))
    for tr, te in skf.split(X, y):
        m = make(params)
        if isinstance(m, XGBClassifier):
            m.fit(X[tr], y[tr], sample_weight=sample_weights(y[tr], cw_mode))
        else:
            m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    return oof

def best_weight(px, pc, y):
    best_w, best_f1 = 0.5, -1
    for w in np.arange(0.0, 1.0001, 0.02):
        f1 = f1_score(y, (w * px + (1 - w) * pc).argmax(1), average="macro")
        if f1 > best_f1:
            best_w, best_f1 = float(w), float(f1)
    return best_w, best_f1

def tune_multipliers(probs, y, passes=4):
    """Coordinate ascent on per-class probability multipliers."""
    mult = np.ones(3)
    cur = f1_score(y, (probs * mult).argmax(1), average="macro")
    grid = np.arange(0.5, 2.051, 0.05)
    for _ in range(passes):
        improved = False
        for c in range(3):
            for g in grid:
                trial = mult.copy(); trial[c] = g
                f1 = f1_score(y, (probs * trial).argmax(1), average="macro")
                if f1 > cur + 1e-9:
                    mult, cur, improved = trial, f1, True
        if not improved:
            break
    return mult, float(cur)

def metrics_block(y, pred, labels):
    return dict(
        accuracy=round(float(accuracy_score(y, pred)), 4),
        macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
        confusion_matrix=confusion_matrix(y, pred).tolist(),
        classification_report=classification_report(
            y, pred, target_names=labels, digits=3, zero_division=0))

# ---------------------------------------------------------------- run
def run_variant(name, feats, df, y, le, skf):
    X = df[feats].to_numpy(dtype=float)
    print(f"\n=== Variant {name}: {len(feats)} features ===", flush=True)
    xp, x_cv = tune_xgb(X, y, skf)
    print(f"XGBoost tuned. CV macro-F1={x_cv:.4f}", flush=True)
    cp, c_cv = tune_cat(X, y, skf)
    print(f"CatBoost tuned. CV macro-F1={c_cv:.4f}", flush=True)

    x_cw = xp.pop("cw_mode")
    px = oof_probs(make_xgb, xp, x_cw, X, y, skf)
    pc = oof_probs(make_cat, cp, "none", X, y, skf)
    labels = list(le.classes_)

    rx = metrics_block(y, px.argmax(1), labels)
    rc = metrics_block(y, pc.argmax(1), labels)
    w, f1_w = best_weight(px, pc, y)
    pe = w * px + (1 - w) * pc
    re_plain = metrics_block(y, pe.argmax(1), labels)
    mult, f1_t = tune_multipliers(pe, y)
    re_tuned = metrics_block(y, (pe * mult).argmax(1), labels)

    # Refit on full data and persist models + config.
    mx = make_xgb(xp); mx.fit(X, y, sample_weight=sample_weights(y, x_cw))
    mx.save_model(f"model_xgboost_{name}.json")
    mc = make_cat(cp); mc.fit(X, y)
    mc.save_model(f"model_catboost_{name}.cbm")
    return dict(
        features=feats, xgboost_params={**xp, "cw_mode": x_cw},
        xgboost_cv_macro_f1=round(x_cv, 4), catboost_params=cp,
        catboost_cv_macro_f1=round(c_cv, 4), ensemble_weight_xgboost=w,
        ensemble_weight_catboost=round(1 - w, 4),
        class_multipliers={labels[i]: round(float(mult[i]), 2) for i in range(3)},
        xgboost_oof=rx, catboost_oof=rc, ensemble_oof_argmax=re_plain,
        ensemble_oof_threshold_tuned=re_tuned), pe, px, pc, mult

def main():
    df = load_and_clean()
    le = LabelEncoder()
    y = le.fit_transform(df["test_end"].str.lower())
    print("Class mapping:", {c: int(i) for i, c in enumerate(le.classes_)})
    print("Class counts:", {c: int((y == i).sum())
                             for i, c in enumerate(le.classes_)})
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    res = {}
    res["A_full"], peA, pxA, pcA, multA = run_variant(
        "A_full", FEATURES_A, df, y, le, skf)
    res["B_preignition"], peB, _, _, _ = run_variant(
        "B_preignition", FEATURES_B, df, y, le, skf)

    oof = pd.DataFrame({
        "test_id": df["test_id"], "true_label": le.inverse_transform(y),
        "xgb_p_completion": pxA[:, 0], "xgb_p_disruption": pxA[:, 1],
        "xgb_p_extinction": pxA[:, 2], "cat_p_completion": pcA[:, 0],
        "cat_p_disruption": pcA[:, 1], "cat_p_extinction": pcA[:, 2],
        "ens_p_completion": peA[:, 0], "ens_p_disruption": peA[:, 1],
        "ens_p_extinction": peA[:, 2]})
    oof["pred_argmax"] = le.inverse_transform(peA.argmax(1))
    oof["pred_threshold_tuned"] = le.inverse_transform((peA * multA).argmax(1))
    oof.to_csv("oof_predictions.csv", index=False)

    with open("ensemble_config.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved: ensemble_config.json, oof_predictions.csv, model files.")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
