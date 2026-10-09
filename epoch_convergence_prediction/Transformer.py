# chronos_eval_transformer_rf_lodo_lomo_avg.py
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

CSV_PATH = "./data/Transformer.csv"
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

print(f"[INFO] Samples: {len(df)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)} | Const-removed: {len(const_num)}")


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
print_table(cv_df, "=== Chronos - Transformer - Overall Results ===")


if "dataset" not in df.columns:
    raise KeyError("Column 'dataset' is required for LODO.")
y_pred_lodo = leave_one_group_out_oof(pipe, X, y, df["dataset"])
lodo_results = metrics(y, y_pred_lodo)


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
              "=== Chronos (Transformer.csv, LODO, RandomForest) — Learning-Rate Sweeps (Averages) ===")

bs_col = bs_key(pred_lodo)
if bs_col:
    fmt_table(pred_lodo, ["model", "dataset", bs_col],
              "=== Chronos (Transformer.csv, LODO, RandomForest) — Batch-Size Sweeps (Averages) ===")

if "optimizer" in pred_lodo.columns:
    fmt_table(pred_lodo, ["model", "dataset", "optimizer"],
              "=== Chronos (Transformer.csv, LODO, RandomForest) — Optimizer Sweeps (Averages) ===")


lr_col = lr_key(pred_lomo)
if lr_col:
    fmt_table(pred_lomo, ["model", "dataset", lr_col],
              "=== Chronos (Transformer.csv, LOMO, RandomForest) — Learning-Rate Sweeps (Averages) ===")

bs_col = bs_key(pred_lomo)
if bs_col:
    fmt_table(pred_lomo, ["model", "dataset", bs_col],
              "=== Chronos (Transformer.csv, LOMO, RandomForest) — Batch-Size Sweeps (Averages) ===")

if "optimizer" in pred_lomo.columns:
    fmt_table(pred_lomo, ["model", "dataset", "optimizer"],
              "=== Chronos (Transformer.csv, LOMO, RandomForest) — Optimizer Sweeps (Averages) ===")
    
    

pipe.fit(X, y)
joblib.dump(pipe, "transformer_epoch_convergence_model.pkl")

print("\n[INFO] Model saved to 'transformer_epoch_convergence_model.pkl'.")

features_names = pipe["prep"].get_feature_names_out().tolist()
if isinstance(features_names, np.ndarray):
    features_names = features_names.tolist()

features_names = [name.split("__")[-1] for name in features_names]

with open("transformer_epoch_convergence_model_features.json", "w") as f:
    json.dump(features_names, f, indent=2)