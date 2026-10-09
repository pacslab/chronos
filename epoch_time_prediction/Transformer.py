import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from typing import Tuple, List
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except Exception:
    print("Please install: pip install xgboost pandas numpy scikit-learn")
    sys.exit(1)


TIME_CANDIDATES = [
    "avg_batch_time_ms", "iter_time_ms", "iteration_time_ms", "batch_time_ms", "time_ms"
]


LEAKY_COLS = {
    "loader_wait_ms", "compute_ms", "p90_batch_time_ms", "p50_batch_ms", "std_batch_ms",
    "examples_per_sec", "tokens_per_sec"
}


ID_TXT_COLS = {
    "gpu_name", "dataset", "model", "architecture", "arch_family",
    "attention_pattern", "activation", "norm_style", "norm_impl",
    "positional_encoding", "input_shape"
}


CATEGORICAL_GPU_COLS = ["gpu_arch", "Memory Type", "Bus"]


def infer_target(df: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer target from {TIME_CANDIDATES}")

def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.maximum(1e-9, b)

def safe_add_ratio(df: pd.DataFrame, num: str, den: str, out: str):
    if num in df.columns and den in df.columns:
        df[out] = _safe_ratio(df[num].astype(float).values, df[den].astype(float).values)

def engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    safe_add_ratio(df, "flops_train_per_batch", "GPU Memory Bandwidth (GB/s)", "flops_to_bw_ratio")
    for peak_col in ["FP32", "FP16", "FP64"]:
        if peak_col in df.columns:
            safe_add_ratio(df, "flops_train_per_batch", peak_col, f"flops_to_{peak_col}_ratio")

    if "GPU Memory Bandwidth (GB/s)" in df.columns and "Memory (GB)" in df.columns:
        safe_add_ratio(df, "GPU Memory Bandwidth (GB/s)", "Memory (GB)", "bw_per_gb")

    if "arithmetic_intensity_train" in df.columns and "FP32" in df.columns:
        safe_add_ratio(df, "arithmetic_intensity_train", "FP32", "ai_over_fp32")

    if "param_bytes" in df.columns and "param_count" in df.columns:
        denom = np.maximum(1, df["param_count"].astype(float).values)
        df["bytes_per_param"] = df["param_bytes"].astype(float).values / denom

    if "act_bytes_per_seq_proxy" in df.columns and "param_bytes" in df.columns:
        df["acts_to_params"] = _safe_ratio(
            df["act_bytes_per_seq_proxy"].astype(float).values,
            df["param_bytes"].astype(float).values
        )

    if "d_model" in df.columns and "n_heads" in df.columns:
        df["d_head_check"] = _safe_ratio(df["d_model"].astype(float).values,
                                         np.maximum(1.0, df["n_heads"].astype(float).values))

    if "d_ff" in df.columns and "d_model" in df.columns:
        df["ff_multiplier"] = _safe_ratio(df["d_ff"].astype(float).values, df["d_model"].astype(float).values)


    if "batch_size" in df.columns:
        if "src_len" in df.columns and "tgt_len" in df.columns:
            src = df["src_len"].fillna(0).astype(float).values
            tgt = df["tgt_len"].fillna(0).astype(float).values
            seq_tokens = src + tgt
        elif "seq_len" in df.columns:
            seq_tokens = df["seq_len"].fillna(0).astype(float).values
        else:
            seq_tokens = np.zeros(len(df), dtype=float)
        df["seq_token_load"] = seq_tokens * df["batch_size"].astype(float).values


    for col in CATEGORICAL_GPU_COLS:
        if col in df.columns and df[col].notna().any():
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    return df

def build_feature_matrix(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    drop = set([target]) | LEAKY_COLS | ID_TXT_COLS | set(CATEGORICAL_GPU_COLS)
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("No numeric feature columns after filtering.")
    X = df[feat_cols].copy()
    y = df[target].astype(float).values
    return X, y, feat_cols


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(1e-9, np.abs(y_true))
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def make_model(seed: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
        tree_method="hist",
    )


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, formatters={
        "Avg_Actual_ms": lambda v: f"{v:.2f}",
        "MAE":           lambda v: f"{v:.3f}",
        "MAPE_%":        lambda v: f"{v:.2f}%",
        "RMSE":          lambda v: f"{v:.3f}",
        "N_test":        lambda v: f"{int(v)}",
    })


def evaluate_80_10_10(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    """Random 80/10/10 split; report metrics on the 10% test set."""
    X_rem, X_te, y_rem, y_te, meta_rem, meta_te = train_test_split(
        X, y, df[["model", "gpu_name"]], test_size=0.10, random_state=seed, shuffle=True
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_rem, y_rem, test_size=(1/9), random_state=seed, shuffle=True
    )

    model = make_model(seed)
    model.fit(X_tr, y_tr)
    
    
    model.save_model("transformer_epoch_time_model.json")
    
    
    yhat_te = model.predict(X_te)

    eval_df = meta_te.copy()
    eval_df["Actual_ms"] = y_te
    eval_df["Pred_ms"] = yhat_te


    rows = []
    for m, g in eval_df.groupby("model"):
        ya, yp = g["Actual_ms"].values, g["Pred_ms"].values
        rows.append({
            "Model": m,
            "N_test": int(len(g)),
            "Avg_Actual_ms": float(np.mean(ya)),
            "MAE": mean_absolute_error(ya, yp),
            "MAPE_%": mape(ya, yp),
            "RMSE": rmse(ya, yp),
        })
    rows.append({
        "Model": "ALL",
        "N_test": int(len(eval_df)),
        "Avg_Actual_ms": float(eval_df["Actual_ms"].mean()),
        "MAE": mean_absolute_error(eval_df["Actual_ms"], eval_df["Pred_ms"]),
        "MAPE_%": mape(eval_df["Actual_ms"], eval_df["Pred_ms"]),
        "RMSE": rmse(eval_df["Actual_ms"], eval_df["Pred_ms"]),
    })
    out_model = pd.DataFrame(rows).sort_values(["Model"])
    subset = out_model[out_model["Model"].isin(["DeiT-Tiny", "ViT"])]
    print("\n=== 80/10/10 — Selected Models (on 10% test) ===")
    print(_fmt(subset))


    rows_b = []
    for (m, gname), g in eval_df.groupby(["model", "gpu_name"]):
        ya, yp = g["Actual_ms"].values, g["Pred_ms"].values
        rows_b.append({
            "Model": m,
            "GPU": str(gname),
            "N_test": int(len(g)),
            "Avg_Actual_ms": float(np.mean(ya)),
            "MAE": mean_absolute_error(ya, yp),
            "MAPE_%": mape(ya, yp),
            "RMSE": rmse(ya, yp),
        })
    out_model_gpu = pd.DataFrame(rows_b).sort_values(["Model", "GPU"])
    print("\n=== 80/10/10 — Per-Model × GPU (on 10% test) ===")
    if not out_model_gpu.empty:
        print(_fmt(out_model_gpu))
    else:
        print("[INFO] Not enough test samples per (Model, GPU) to print this table.")

def evaluate_lodo(df_all: pd.DataFrame, X_all: pd.DataFrame, y_all: np.ndarray, seed: int = 42):

    if "dataset" not in df_all.columns:
        print("\n[WARN] LODO skipped (no 'dataset' column).")
        return
    results = []
    for ds, idx_te in df_all.groupby("dataset").groups.items():
        idx_te = list(idx_te)
        idx_tr = [i for i in range(len(df_all)) if i not in idx_te]
        if len(idx_tr) == 0 or len(idx_te) == 0:
            continue

        X_tr, y_tr = X_all.iloc[idx_tr], y_all[idx_tr]
        X_te, y_te = X_all.iloc[idx_te], y_all[idx_te]

        model = make_model(seed)
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_te)

        results.append({
            "Dataset_LeftOut": ds,
            "N_test": int(len(y_te)),
            "Avg_Actual_ms": float(np.mean(y_te)),
            "MAE": float(mean_absolute_error(y_te, yhat)),
            "MAPE_%": float(mape(y_te, yhat)),
            "RMSE": float(rmse(y_te, yhat)),
        })
    if not results:
        print("\n[WARN] No LODO results.")
        return
    out = pd.DataFrame(results).sort_values("Dataset_LeftOut")
    overall = {
        "Dataset_LeftOut": "ALL",
        "N_test": int(out["N_test"].sum()),
        "Avg_Actual_ms": float(np.average(out["Avg_Actual_ms"], weights=out["N_test"])),
        "MAE": float(np.average(out["MAE"], weights=out["N_test"])),
        "MAPE_%": float(np.average(out["MAPE_%"], weights=out["N_test"])),
        "RMSE": float(np.average(out["RMSE"], weights=out["N_test"])),
    }
    out = pd.concat([out, pd.DataFrame([overall])], ignore_index=True)
    print("\n=== LODO — Leave-One-Dataset-Out (test on held-out dataset) ===")
    print(_fmt(out))

def evaluate_lomo(df_all: pd.DataFrame, X_all: pd.DataFrame, y_all: np.ndarray, seed: int = 42):

    if "model" not in df_all.columns:
        print("\n[WARN] LOMO skipped (no 'model' column).")
        return
    results = []
    for m, idx_te in df_all.groupby("model").groups.items():
        idx_te = list(idx_te)
        idx_tr = [i for i in range(len(df_all)) if i not in idx_te]
        if len(idx_tr) == 0 or len(idx_te) == 0:
            continue

        X_tr, y_tr = X_all.iloc[idx_tr], y_all[idx_tr]
        X_te, y_te = X_all.iloc[idx_te], y_all[idx_te]

        model = make_model(seed)
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_te)

        results.append({
            "Model_LeftOut": m,
            "N_test": int(len(y_te)),
            "Avg_Actual_ms": float(np.mean(y_te)),
            "MAE": float(mean_absolute_error(y_te, yhat)),
            "MAPE_%": float(mape(y_te, yhat)),
            "RMSE": float(rmse(y_te, yhat)),
        })
    if not results:
        print("\n[WARN] No LOMO results.")
        return
    out = pd.DataFrame(results).sort_values("Model_LeftOut")
    overall = {
        "Model_LeftOut": "ALL",
        "N_test": int(out["N_test"].sum()),
        "Avg_Actual_ms": float(np.average(out["Avg_Actual_ms"], weights=out["N_test"])),
        "MAE": float(np.average(out["MAE"], weights=out["N_test"])),
        "MAPE_%": float(np.average(out["MAPE_%"], weights=out["N_test"])),
        "RMSE": float(np.average(out["RMSE"], weights=out["N_test"])),
    }
    out = pd.concat([out, pd.DataFrame([overall])], ignore_index=True)
    print("\n=== LOMO — Leave-One-Model-Out (test on held-out model) ===")
    print(_fmt(out))


def main():
    ap = argparse.ArgumentParser("Transformer Timing Evaluation (80/10/10, LODO, LOMO)")
    ap.add_argument("--csv", type=str, default="./data/Transformer.csv", help="Path to Transformer CSV.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-lodo", action="store_true", help="Skip LODO evaluation.")
    ap.add_argument("--skip-lomo", action="store_true", help="Skip LOMO evaluation.")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    target = infer_target(df)
    req = [target, "batch_size", "model", "gpu_name"]
    for c in req:
        if c not in df.columns:
            sys.exit(f"ERROR: missing required column: {c}")

    df = df.dropna(subset=req).copy()


    df = engineer_derived_features(df)
    X, y, feat_cols = build_feature_matrix(df, target)
    print(f"[INFO] Features: {len(feat_cols)} numeric columns after filtering/leak removal.")
    
    print(X.columns.tolist(), len(X.columns.tolist()))


    evaluate_80_10_10(df, X, y, seed=args.seed)
    
    model = make_model(seed=args.seed)
    model.fit(X, y)
    model.save_model("transformer_epoch_time_model.json")
    
    features_names = model.get_booster().feature_names
    with open("transformer_epoch_time_model_features.json", "w") as f:
        json.dump(features_names, f)


if __name__ == "__main__":
    main()