"""Full CV pipeline: builds features, evaluates LightGBM / XGBoost / logistic
regression under both the walk-forward (leakage-safe) and shuffled-date
(diagnostic) CV schemes, and reports accuracy + feature importances.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw, add_ts_int
from src.features import build_features, get_categorical_features, get_linear_feature_columns
from src.cv import walk_forward_splits, shuffled_date_kfold_splits
from src.models import fit_lgbm, predict_lgbm, fit_xgb, predict_xgb, fit_logreg, predict_logreg, ensemble_average

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def evaluate_cv(feat_df, y, ts_int, splits, scheme_name, run_logreg=True):
    cat_features = get_categorical_features()
    lin_cols = get_linear_feature_columns(feat_df)

    rows = []
    importances = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_tr_bin = (y_tr > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)

        t0 = time.time()
        lgbm_model = fit_lgbm(X_tr, y_tr_bin, X_val, y_val_bin, categorical_features=cat_features)
        lgbm_pred = predict_lgbm(lgbm_model, X_val)
        lgbm_acc = accuracy_score(y_val_bin, (lgbm_pred > 0.5).astype(int))
        importances.append(pd.Series(lgbm_model.feature_importance(importance_type="gain"), index=feat_df.columns))

        xgb_model = fit_xgb(X_tr, y_tr_bin, X_val, y_val_bin)
        xgb_pred = predict_xgb(xgb_model, X_val)
        xgb_acc = accuracy_score(y_val_bin, (xgb_pred > 0.5).astype(int))

        row = {"scheme": scheme_name, "fold": fold_i, "n_train": len(train_idx), "n_val": len(val_idx),
               "lgbm_acc": lgbm_acc, "xgb_acc": xgb_acc}

        if run_logreg:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            grp_tr = ohe.fit_transform(X_tr[["GROUP"]])
            grp_val = ohe.transform(X_val[["GROUP"]])
            X_tr_lin = np.hstack([X_tr[lin_cols].to_numpy(dtype=float), grp_tr])
            X_val_lin = np.hstack([X_val[lin_cols].to_numpy(dtype=float), grp_val])
            logreg_model = fit_logreg(X_tr_lin, y_tr_bin)
            logreg_pred = predict_logreg(logreg_model, X_val_lin)
            logreg_acc = accuracy_score(y_val_bin, (logreg_pred > 0.5).astype(int))
            row["logreg_acc"] = logreg_acc

            ens_pred = ensemble_average(lgbm_pred, xgb_pred, logreg_pred)
            row["ensemble_acc"] = accuracy_score(y_val_bin, (ens_pred > 0.5).astype(int))
        else:
            ens_pred = ensemble_average(lgbm_pred, xgb_pred)
            row["ensemble_acc"] = accuracy_score(y_val_bin, (ens_pred > 0.5).astype(int))

        row["seconds"] = time.time() - t0
        rows.append(row)
        print(f"[{scheme_name}] fold {fold_i}: n_train={len(train_idx)} n_val={len(val_idx)} "
              f"lgbm={lgbm_acc:.4f} xgb={xgb_acc:.4f} "
              f"logreg={row.get('logreg_acc', float('nan')):.4f} ens={row['ensemble_acc']:.4f} "
              f"({row['seconds']:.1f}s)")

    results = pd.DataFrame(rows)
    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return results, mean_importance


def main():
    X_train_raw, y_train, X_test_raw, _ = load_raw("data")
    X_train_raw = add_ts_int(X_train_raw)

    print("Building features...")
    feat_train = build_features(X_train_raw)
    print(f"feat_train shape: {feat_train.shape}")

    ts_int = X_train_raw["TS_INT"].to_numpy()

    print("\n" + "=" * 70)
    print("PRIMARY: walk-forward (leakage-safe) CV")
    print("=" * 70)
    wf_splits = walk_forward_splits(ts_int, n_splits=6, purge=5)
    wf_results, wf_importance = evaluate_cv(feat_train, y_train, ts_int, wf_splits, "walk_forward")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: shuffled date KFold (matches benchmark's scheme)")
    print("=" * 70)
    sh_splits = shuffled_date_kfold_splits(ts_int, n_splits=8, seed=0)
    sh_results, sh_importance = evaluate_cv(feat_train, y_train, ts_int, sh_splits, "shuffled_kfold")

    print("\n" + "=" * 70)
    print("SUMMARY (mean accuracy across folds)")
    print("=" * 70)
    for name, res in [("walk_forward", wf_results), ("shuffled_kfold", sh_results)]:
        means = res[[c for c in res.columns if c.endswith("_acc")]].mean()
        print(f"\n{name}:")
        print(means.to_string())

    wf_results.to_csv(ARTIFACTS / "cv_results_walk_forward.csv", index=False)
    sh_results.to_csv(ARTIFACTS / "cv_results_shuffled_kfold.csv", index=False)
    wf_importance.to_csv(ARTIFACTS / "feature_importance_walk_forward.csv", header=["gain"])
    sh_importance.to_csv(ARTIFACTS / "feature_importance_shuffled_kfold.csv", header=["gain"])

    print("\nTop 20 features by gain (walk-forward folds):")
    print(wf_importance.head(20).to_string())


if __name__ == "__main__":
    main()
