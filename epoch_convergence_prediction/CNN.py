import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import warnings
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import KFold, cross_val_predict, LeaveOneGroupOut
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = "./data/CNN.csv"
TARGET = "T_conv"
RANDOM_STATE = 42

def print_table(df: pd.DataFrame, title: str):
    print(f"\n{title}")
    print("-" * max(70, len(title)))
    df2 = df.copy()
    for c in df2.select_dtypes(include=[float]).columns:
        df2[c] = df2[c].map(lambda x: f"{x:.4f}")
    print(df2.to_string(index=False))

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0.0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}

def leave_one_group_out_oof(pipe, X: pd.DataFrame, y: np.ndarray, groups: pd.Series):
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y, dtype=float)
    for tr, te in logo.split(X, y, groups=groups):
        pipe.fit(X.iloc[tr], y[tr])
        y_pred[te] = pipe.predict(X.iloc[te])
    return y_pred

df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
if TARGET not in df.columns:
    raise KeyError(f"Target '{TARGET}' not found. Columns: {list(df.columns)}")
df = df[~df[TARGET].isna()].reset_index(drop=True)

forbidden = {TARGET, "T_80close", "T_90close"}
feature_cols = [c for c in df.columns if c not in forbidden]

num_cols, cat_cols = [], []
for c in feature_cols:
    (num_cols if pd.api.types.is_numeric_dtype(df[c]) else cat_cols).append(c)

const_num = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
num_cols = [c for c in num_cols if c not in const_num]

preproc = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ],
    remainder="drop"
)

X = df[feature_cols].copy()
y = df[TARGET].astype(float).to_numpy()

print(f"[INFO] Samples: {len(df)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

rf = RandomForestRegressor(
    n_estimators=400,
    n_jobs=1,
    random_state=RANDOM_STATE
)

pipe = Pipeline([("prep", preproc), ("reg", rf)])

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_pred_cv = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
cv_results = metrics(y, y_pred_cv)
cv_df = pd.DataFrame([{"Regressor": "RandomForest", **cv_results}])

if "dataset" not in df.columns:
    raise KeyError("Column 'dataset' is required for LODO.")
y_pred_lodo = leave_one_group_out_oof(pipe, X, y, df["dataset"])
lodo_results = metrics(y, y_pred_lodo)
lodo_df = pd.DataFrame([{"Regressor": "RandomForest", **lodo_results}])

if "model" not in df.columns:
    raise KeyError("Column 'model' is required for LOMO.")
y_pred_lomo = leave_one_group_out_oof(pipe, X, y, df["model"])
lomo_results = metrics(y, y_pred_lomo)
lomo_df = pd.DataFrame([{"Regressor": "RandomForest", **lomo_results}])

def fmt_table(df_in: pd.DataFrame, group_keys: list, title: str):
    cols_exist = [c for c in group_keys if c in df_in.columns]
    if len(cols_exist) != len(group_keys):
        missing = [c for c in group_keys if c not in df_in.columns]
        print(f"\n[WARN] Skipping {title} — missing columns: {missing}")
        return
    g = (
        df_in.groupby(cols_exist, as_index=False)[["true_epochs", "pred_epochs"]]
        .mean()
        .rename(columns={"true_epochs": "true_epochs_avg", "pred_epochs": "pred_epochs_avg"})
        .sort_values(by=cols_exist)
        .reset_index(drop=True)
    )
    print_table(g, title)

pred_lodo = df.copy()
pred_lodo["true_epochs"] = y
pred_lodo["pred_epochs"] = y_pred_lodo

pred_lomo = df.copy()
pred_lomo["true_epochs"] = y
pred_lomo["pred_epochs"] = y_pred_lomo

def lr_key(df_):
    return "learning_rate" if "learning_rate" in df_.columns else (
        "learning_rate_log10" if "learning_rate_log10" in df_.columns else None
    )

def bs_key(df_):
    return "batch_size" if "batch_size" in df_.columns else (
        "batch_size_log10" if "batch_size_log10" in df_.columns else None
    )


lr_col = lr_key(pred_lodo)
if lr_col:
    fmt_table(pred_lodo, ["model", "dataset", lr_col],
              "=== Chronos (CNN.csv, LODO, RandomForest) — Learning-Rate Sweeps (Averages) ===")

bs_col = bs_key(pred_lodo)
if bs_col:
    fmt_table(pred_lodo, ["model", "dataset", bs_col],
              "=== Chronos (CNN.csv, LODO, RandomForest) — Batch-Size Sweeps (Averages) ===")

if "optimizer" in pred_lodo.columns:
    fmt_table(pred_lodo, ["model", "dataset", "optimizer"],
              "=== Chronos (CNN.csv, LODO, RandomForest) — Optimizer Sweeps (Averages) ===")


lr_col = lr_key(pred_lomo)
if lr_col:
    fmt_table(pred_lomo, ["model", "dataset", lr_col],
              "=== Chronos (CNN.csv, LOMO, RandomForest) — Learning-Rate Sweeps (Averages) ===")

bs_col = bs_key(pred_lomo)
if bs_col:
    fmt_table(pred_lomo, ["model", "dataset", bs_col],
              "=== Chronos (CNN.csv, LOMO, RandomForest) — Batch-Size Sweeps (Averages) ===")

if "optimizer" in pred_lomo.columns:
    fmt_table(pred_lomo, ["model", "dataset", "optimizer"],
              "=== Chronos (CNN.csv, LOMO, RandomForest) — Optimizer Sweeps (Averages) ===")


import numpy as _np
from sklearn.metrics import mean_absolute_error, r2_score

def _rmse(y_true, y_pred):
    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    return float(_np.sqrt(_np.mean((y_true - y_pred) ** 2)))

def _median_ae(y_true, y_pred):
    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    return float(_np.median(_np.abs(y_true - y_pred)))

def _mape_pct(y_true, y_pred, eps=1e-8):
    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    denom = _np.maximum(_np.abs(y_true), eps)
    return float(_np.mean(_np.abs((y_true - y_pred) / denom)) * 100.0)

def _bre(y_true, y_pred):
    """Bounded Relative Error: mean(min(|err|/max(|y|,1), 1))."""
    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    denom = _np.maximum(_np.abs(y_true), 1.0)
    rel = _np.abs(y_true - y_pred) / denom
    return float(_np.mean(_np.minimum(rel, 1.0)))

def _avg_accuracy(y_true, y_pred):
    """1 - BRE, averaged (clipped per-sample to [0,1])."""
    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    denom = _np.maximum(_np.abs(y_true), 1.0)
    rel = _np.abs(y_true - y_pred) / denom
    acc = 1.0 - _np.minimum(rel, 1.0)
    return float(_np.mean(_np.maximum(_np.minimum(acc, 1.0), 0.0)))

def _full_metrics(y_true, y_pred):
    return {
        "N": int(len(y_true)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "MedianAE": _median_ae(y_true, y_pred),
        "MAPE%": _mape_pct(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "BRE": _bre(y_true, y_pred),
        "AvgAccuracy": _avg_accuracy(y_true, y_pred),
    }

overall_rows = []
overall_rows.append({"Regime": "CV (5-fold)", "Regressor": "RandomForest", **_full_metrics(y, y_pred_cv)})
overall_rows.append({"Regime": "LODO",        "Regressor": "RandomForest", **_full_metrics(y, y_pred_lodo)})
overall_rows.append({"Regime": "LOMO",        "Regressor": "RandomForest", **_full_metrics(y, y_pred_lomo)})

overall_df = pd.DataFrame(overall_rows)
overall_df = overall_df[["Regime","Regressor","N","MAE","RMSE","MedianAE","MAPE%","R2","BRE","AvgAccuracy"]]
print_table(
    overall_df.sort_values(["MAE","RMSE"]).reset_index(drop=True),
    "=== Chronos (CNN.csv) — Overall Evaluation (CV/LODO/LOMO) — Random Forest ==="
)


def _group_metrics(df_pred: pd.DataFrame, group_col: str, title: str):
    if group_col not in df_pred.columns:
        print(f"\n[WARN] Skipping {title} — missing '{group_col}' column.")
        return
    rows = []
    for g, dd in df_pred.groupby(group_col):
        rows.append({
            group_col: g,
            **_full_metrics(dd["true_epochs"].values, dd["pred_epochs"].values)
        })
    out = pd.DataFrame(rows)
    out = out[[group_col,"N","MAE","RMSE","MedianAE","MAPE%","R2","BRE","AvgAccuracy"]] \
             .sort_values(["MAE","RMSE"]).reset_index(drop=True)
    print_table(out, title)

_group_metrics(pred_lodo, "dataset",
               "=== Chronos (CNN.csv) — LODO Per-Dataset Diagnostics (RandomForest) ===")
_group_metrics(pred_lomo, "model",
               "=== Chronos (CNN.csv) — LOMO Per-Model Diagnostics (RandomForest) ===")

pipe.fit(X, y)
joblib.dump(pipe, "cnn_epoch_convergence_model.pkl")

print("\n[INFO] Model saved to 'cnn_epoch_convergence_model.pkl'.")

features_names = pipe["prep"].get_feature_names_out().tolist()
if isinstance(features_names, np.ndarray):
    features_names = features_names.tolist()

features_names = [name.split("__")[-1] for name in features_names]

with open("cnn_epoch_convergence_model_features.json", "w") as f:
    json.dump(features_names, f, indent=2)