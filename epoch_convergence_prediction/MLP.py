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

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = "./data/MLP.csv"
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
    return {"MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}

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
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ],
    remainder="drop"
)

X = df[feature_cols].copy()
y = df[TARGET].astype(float).to_numpy()

def has(col): return col in df.columns

print(f"[INFO] Samples: {len(df)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "SVR(RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=400, n_jobs=1, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=600, n_jobs=1, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows, cv_oof = [], {}

for name, reg in models.items():
    pipe = Pipeline([("prep", preproc), ("reg", reg)])
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    cv_oof[name] = y_pred
    cv_rows.append({"Regressor": name, **metrics(y, y_pred)})

cv_df = pd.DataFrame(cv_rows).sort_values(["MAE", "R2"], ascending=[True, False]).reset_index(drop=True)
print_table(cv_df, "=== Chronos — In-Distribution (5-fold CV) — MAE & R² ===")
best_cv_name = cv_df.iloc[0]["Regressor"]

if not has("dataset"):
    raise KeyError("Column 'dataset' is required for LODO.")
lodo_rows, lodo_oof = [], {}

for name, reg in models.items():
    pipe = Pipeline([("prep", preproc), ("reg", reg)])
    y_pred = leave_one_group_out_oof(pipe, X, y, df["dataset"])
    lodo_oof[name] = y_pred
    lodo_rows.append({"Regressor": name, **metrics(y, y_pred)})

lodo_df = pd.DataFrame(lodo_rows).sort_values(["MAE", "R2"], ascending=[True, False]).reset_index(drop=True)
print_table(lodo_df, "=== Chronos — Leave-One-Dataset-Out (LODO) — MAE & R² ===")
lodo_best_name = lodo_df.iloc[0]["Regressor"]
lodo_yhat = lodo_oof[lodo_best_name]


if not has("model"):
    raise KeyError("Column 'model' is required for LOMO.")
lomo_rows, lomo_oof = [], {}

for name, reg in models.items():
    pipe = Pipeline([("prep", preproc), ("reg", reg)])
    y_pred = leave_one_group_out_oof(pipe, X, y, df["model"])
    lomo_oof[name] = y_pred
    lomo_rows.append({"Regressor": name, **metrics(y, y_pred)})

lomo_df = pd.DataFrame(lomo_rows).sort_values(["MAE", "R2"], ascending=[True, False]).reset_index(drop=True)
print_table(lomo_df, "=== Chronos — Leave-One-Model-Out (LOMO) — MAE & R² ===")
lomo_best_name = lomo_df.iloc[0]["Regressor"]
lomo_yhat = lomo_oof[lomo_best_name]


def fmt_table(df_in: pd.DataFrame, group_keys: list, title: str):
    """
    Group by group_keys and print average true/pred epochs once per (model, dataset, key).
    """
    cols_exist = [c for c in group_keys if c in df_in.columns]
    if len(cols_exist) != len(group_keys):
        missing = [c for c in group_keys if c not in df_in.columns]
        print(f"\n[WARN] Skipping {title} — missing columns: {missing}")
        return
    g = (df_in
         .groupby(cols_exist, as_index=False)[["true_epochs", "pred_epochs"]]
         .mean()
         .rename(columns={"true_epochs": "true_epochs_avg", "pred_epochs": "pred_epochs_avg"}))

    g = g.sort_values(by=cols_exist).reset_index(drop=True)
    print_table(g, title)


pred_lodo = df.copy()
pred_lodo["true_epochs"] = y
pred_lodo["pred_epochs"] = lodo_yhat

pred_lomo = df.copy()
pred_lomo["true_epochs"] = y
pred_lomo["pred_epochs"] = lomo_yhat


def lr_key(df_):
    return "learning_rate" if "learning_rate" in df_.columns else ("learning_rate_log10" if "learning_rate_log10" in df_.columns else None)
def bs_key(df_):
    return "batch_size" if "batch_size" in df_.columns else ("batch_size_log10" if "batch_size_log10" in df_.columns else None)


lr_col = lr_key(pred_lodo)
if lr_col:
    fmt_table(pred_lodo, ["model", "dataset", lr_col],
              f"=== Chronos (LODO, {lodo_best_name}) — Learning-Rate Sweeps (Averages) ===")
else:
    print("\n[WARN] No learning rate column found for LODO table.")

bs_col = bs_key(pred_lodo)
if bs_col:
    fmt_table(pred_lodo, ["model", "dataset", bs_col],
              f"=== Chronos (LODO, {lodo_best_name}) — Batch-Size Sweeps (Averages) ===")
else:
    print("\n[WARN] No batch-size column found for LODO table.")

if "optimizer" in pred_lodo.columns:
    fmt_table(pred_lodo, ["model", "dataset", "optimizer"],
              f"=== Chronos (LODO, {lodo_best_name}) — Optimizer Sweeps (Averages) ===")
else:
    print("\n[WARN] No optimizer column found for LODO table.")


lr_col = lr_key(pred_lomo)
if lr_col:
    fmt_table(pred_lomo, ["model", "dataset", lr_col],
              f"=== Chronos (LOMO, {lomo_best_name}) — Learning-Rate Sweeps (Averages) ===")
else:
    print("\n[WARN] No learning rate column found for LOMO table.")

bs_col = bs_key(pred_lomo)
if bs_col:
    fmt_table(pred_lomo, ["model", "dataset", bs_col],
              f"=== Chronos (LOMO, {lomo_best_name}) — Batch-Size Sweeps (Averages) ===")
else:
    print("\n[WARN] No batch-size column found for LOMO table.")

if "optimizer" in pred_lomo.columns:
    fmt_table(pred_lomo, ["model", "dataset", "optimizer"],
              f"=== Chronos (LOMO, {lomo_best_name}) — Optimizer Sweeps (Averages) ===")
else:
    print("\n[WARN] No optimizer column found for LOMO table.")


import numpy as _np

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

    y_true, y_pred = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    denom = _np.maximum(_np.abs(y_true), 1.0)
    rel = _np.abs(y_true - y_pred) / denom
    return float(_np.mean(_np.minimum(rel, 1.0)))

def _avg_accuracy(y_true, y_pred):

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


cv_best_yhat   = cv_oof[best_cv_name]
lodo_best_yhat = lodo_oof[lodo_best_name]
lomo_best_yhat = lomo_oof[lomo_best_name]

overall_rows = []
overall_rows.append({"Regime": "CV (5-fold)", "BestModel": best_cv_name,   **_full_metrics(y, cv_best_yhat)})
overall_rows.append({"Regime": "LODO",        "BestModel": lodo_best_name, **_full_metrics(y, lodo_best_yhat)})
overall_rows.append({"Regime": "LOMO",        "BestModel": lomo_best_name, **_full_metrics(y, lomo_best_yhat)})

overall_df = pd.DataFrame(overall_rows)


_order = ["Regime", "BestModel", "N", "MAE", "RMSE", "MedianAE", "MAPE%", "R2", "BRE", "AvgAccuracy"]
overall_df = overall_df[_order]

print_table(
    overall_df.sort_values(["MAE", "RMSE"]).reset_index(drop=True),
    "=== Chronos — Overall Evaluation (Best Model per Regime) ==="
)

def _evaluate_regressor_across_regimes(name: str):
    rows = []
    rows.append({"Regime": "CV (5-fold)", "Regressor": name, **_full_metrics(y, cv_oof[name])})
    rows.append({"Regime": "LODO",        "Regressor": name, **_full_metrics(y, lodo_oof[name])})
    rows.append({"Regime": "LOMO",        "Regressor": name, **_full_metrics(y, lomo_oof[name])})
    return pd.DataFrame(rows)

all_regs = sorted(models.keys())
big = pd.concat([_evaluate_regressor_across_regimes(r) for r in all_regs], ignore_index=True)


agg = (big
       .groupby("Regressor", as_index=False)
       .agg({"N":"first","MAE":"mean","RMSE":"mean","MedianAE":"mean","MAPE%":"mean","R2":"mean","BRE":"mean","AvgAccuracy":"mean"})
       .sort_values(["MAE","RMSE"], ascending=[True, True])
       .reset_index(drop=True))

print_table(agg, "=== Chronos — Overall Regressor Leaderboard (mean over CV/LODO/LOMO) ===")

best_model_name = agg.iloc[0]["Regressor"]
print(f"\n[INFO] Best overall regressor: {best_model_name}")


best_reg = models[best_model_name]
best_pipe = Pipeline([("prep", preproc), ("reg", best_reg)])


print("[INFO] Training best pipeline on full dataset...")
best_pipe.fit(X, y)

joblib.dump(best_pipe, "mlp_epoch_convergence_model.pkl")

print("\n[INFO] Model saved to 'mlp_epoch_convergence_model.pkl'.")

features_names = best_pipe["prep"].get_feature_names_out().tolist()
if isinstance(features_names, np.ndarray):
    features_names = features_names.tolist()

features_names = [name.split("__")[-1] for name in features_names]

with open("mlp_epoch_convergence_model_features.json", "w") as f:
    json.dump(features_names, f, indent=2)
