"""Feature-group ablation: how much does each engineered feature family add
on top of the raw columns, under leakage-safe walk-forward CV?

Uses LightGBM only (fastest, and the importance ranking already comes from
it) with a smaller n_splits/num_boost_round than the main training run to
keep this a quick diagnostic rather than a full model search.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw, add_ts_int
from src.features import build_features, get_categorical_features
from src.cv import walk_forward_splits
from src.models import fit_lgbm, predict_lgbm

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

RAW_COLS = (
    [f"RET_{i}" for i in range(1, 21)]
    + [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
    + ["MEDIAN_DAILY_TURNOVER", "GROUP", "ALLOCATION"]
)

FEATURE_GROUPS = {
    "momentum": ["ret1_minus_older_mean", "ret1_zscore_vs_older", "avg_perf_3", "avg_perf_5",
                 "avg_perf_10", "avg_perf_15", "avg_perf_20", "std_perf_20", "ret_autocorr_lag1"],
    "distribution_shape": ["ret_skew", "ret_kurt", "vol_skew", "vol_kurt"],
    "trend": ["ret_trend_slope", "ret_diff_mean", "ret_diff_std"],
    "volume_interaction": ["ret_vol_corr", "sign_divergence", "volume_1_available",
                            "signed_volume_2", "vol_std_20"],
    "turnover_regime": ["turnover", "turnover_rank_group", "turnover_rank_date", "turnover_x_ret1"],
    "cross_sectional": ["ret1_minus_group_mean", "ret1_minus_date_mean", "ret1_pct_rank_date",
                         "avg_perf5_minus_group_mean"],
}


def eval_columns(feat_df, y, ts_int, cols, cat_features, n_splits=4, purge=5, num_boost_round=400):
    splits = walk_forward_splits(ts_int, n_splits=n_splits, purge=purge)
    accs = []
    for train_idx, val_idx in splits:
        X_tr, X_val = feat_df.iloc[train_idx][cols], feat_df.iloc[val_idx][cols]
        y_tr = (y.iloc[train_idx] > 0).astype(int)
        y_val = (y.iloc[val_idx] > 0).astype(int)
        model = fit_lgbm(X_tr, y_tr, X_val, y_val, categorical_features=cat_features,
                          num_boost_round=num_boost_round)
        pred = predict_lgbm(model, X_val)
        accs.append(accuracy_score(y_val, (pred > 0.5).astype(int)))
    return sum(accs) / len(accs), accs


def main():
    X_train_raw, y_train, _, _ = load_raw("data")
    X_train_raw = add_ts_int(X_train_raw)
    feat_train = build_features(X_train_raw)
    ts_int = X_train_raw["TS_INT"].to_numpy()
    cat_features = [c for c in get_categorical_features() if c in RAW_COLS]

    rows = []

    t0 = time.time()
    raw_acc, raw_accs = eval_columns(feat_train, y_train, ts_int, RAW_COLS, cat_features)
    print(f"raw_only: mean_acc={raw_acc:.4f} folds={['%.4f' % a for a in raw_accs]} ({time.time()-t0:.1f}s)")
    rows.append({"config": "raw_only", "mean_acc": raw_acc, "delta_vs_raw": 0.0})

    cumulative_cols = list(RAW_COLS)
    for group_name, group_cols in FEATURE_GROUPS.items():
        cumulative_cols = cumulative_cols + [c for c in group_cols if c in feat_train.columns]
        t0 = time.time()
        acc, accs = eval_columns(feat_train, y_train, ts_int, cumulative_cols, cat_features)
        delta = acc - raw_acc
        print(f"+ {group_name}: mean_acc={acc:.4f} (delta vs raw={delta:+.4f}) "
              f"folds={['%.4f' % a for a in accs]} ({time.time()-t0:.1f}s)")
        rows.append({"config": f"+{group_name}", "mean_acc": acc, "delta_vs_raw": delta})

    all_cols = [c for c in feat_train.columns]
    t0 = time.time()
    full_acc, full_accs = eval_columns(feat_train, y_train, ts_int, all_cols, cat_features)
    print(f"full (all features): mean_acc={full_acc:.4f} (delta vs raw={full_acc-raw_acc:+.4f}) "
          f"folds={['%.4f' % a for a in full_accs]} ({time.time()-t0:.1f}s)")
    rows.append({"config": "full", "mean_acc": full_acc, "delta_vs_raw": full_acc - raw_acc})

    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACTS / "ablation_results.csv", index=False)
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
