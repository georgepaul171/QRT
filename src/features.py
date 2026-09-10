"""Feature engineering for the asset allocation forecasting challenge.

Column convention (confirmed from X_test.csv / benchmark notebook): RET_1 and
SIGNED_VOLUME_1 are the most recent day; RET_20 / SIGNED_VOLUME_20 are 20 days
back. "Chronological" arrays below are reordered oldest -> newest for
trend/autocorrelation calculations where day order matters.
"""
import numpy as np
import pandas as pd

from src.data import RET_COLS, VOL_COLS


def _row_corr(a, b, min_count=5):
    """Pearson correlation per row between two (n, k) arrays, NaN-safe."""
    mask = ~np.isnan(a) & ~np.isnan(b)
    cnt = mask.sum(axis=1)
    a0 = np.where(mask, a, 0.0)
    b0 = np.where(mask, b, 0.0)
    safe_cnt = np.clip(cnt, 1, None)
    mean_a = a0.sum(axis=1) / safe_cnt
    mean_b = b0.sum(axis=1) / safe_cnt
    da = np.where(mask, a - mean_a[:, None], 0.0)
    db = np.where(mask, b - mean_b[:, None], 0.0)
    cov = (da * db).sum(axis=1)
    var_a = (da ** 2).sum(axis=1)
    var_b = (db ** 2).sum(axis=1)
    denom = np.sqrt(var_a * var_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    corr[(denom == 0) | (cnt < min_count)] = np.nan
    return corr


def build_features(df):
    """Build the engineered feature matrix for one block (train or test).

    Cross-sectional aggregates (group/date means, ranks) are computed within
    this dataframe only, using same-day information — this is safe because
    RET_1/SIGNED_VOLUME_1 etc. are already-realized values as of the
    prediction date, not future information.
    """
    df = df.copy()
    feats = {}

    R = df[RET_COLS].to_numpy(dtype=float)  # RET_1 (newest) .. RET_20 (oldest)
    R_chrono = R[:, ::-1]  # oldest -> newest

    older_cols = [f"RET_{i}" for i in range(2, 21)]
    older_mean = df[older_cols].mean(axis=1)
    older_std = df[older_cols].std(axis=1)

    # --- Momentum / mean-reversion ---
    feats["ret_1"] = df["RET_1"]
    feats["ret1_minus_older_mean"] = df["RET_1"] - older_mean
    feats["ret1_zscore_vs_older"] = (df["RET_1"] - older_mean) / older_std.replace(0, np.nan)

    for i in [3, 5, 10, 15, 20]:
        feats[f"avg_perf_{i}"] = df[[f"RET_{k}" for k in range(1, i + 1)]].mean(axis=1)
    feats["std_perf_20"] = df[RET_COLS].std(axis=1)

    a = R_chrono[:, :-1]
    b = R_chrono[:, 1:]
    feats["ret_autocorr_lag1"] = _row_corr(a, b)

    # --- Distribution shape ---
    feats["ret_skew"] = df[RET_COLS].skew(axis=1)
    feats["ret_kurt"] = df[RET_COLS].kurtosis(axis=1)
    feats["vol_skew"] = df[VOL_COLS].skew(axis=1)
    feats["vol_kurt"] = df[VOL_COLS].kurtosis(axis=1)

    # --- Trend / first-difference across the 20-day window ---
    t = np.arange(R_chrono.shape[1], dtype=float)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    feats["ret_trend_slope"] = (R_chrono * t_c).sum(axis=1) / denom

    diffs = np.diff(R_chrono, axis=1)
    feats["ret_diff_mean"] = diffs.mean(axis=1)
    feats["ret_diff_std"] = diffs.std(axis=1)

    # --- Volume-return interaction (SIGNED_VOLUME_1 is ~91% missing on most
    # dates as a structural per-date artifact; days 2-20 are reliably
    # populated, so those are used for anything requiring row-wise pairing) ---
    reliable_ret_cols = [f"RET_{i}" for i in range(2, 21)]
    reliable_vol_cols = [f"SIGNED_VOLUME_{i}" for i in range(2, 21)]
    Rr = df[reliable_ret_cols].to_numpy(dtype=float)
    Vr = df[reliable_vol_cols].to_numpy(dtype=float)
    feats["ret_vol_corr"] = _row_corr(Rr, Vr)

    sign_ret1 = np.sign(df["RET_1"].to_numpy())
    sign_vol2 = np.sign(df["SIGNED_VOLUME_2"].to_numpy())
    feats["sign_divergence"] = (sign_ret1 != sign_vol2).astype(float)

    feats["volume_1_available"] = df["SIGNED_VOLUME_1"].notna().astype(int)
    feats["signed_volume_2"] = df["SIGNED_VOLUME_2"]
    feats["vol_std_20"] = df[VOL_COLS].std(axis=1)

    # --- Turnover as a liquidity/regime indicator ---
    feats["turnover"] = df["MEDIAN_DAILY_TURNOVER"]
    feats["turnover_rank_group"] = df.groupby("GROUP")["MEDIAN_DAILY_TURNOVER"].rank(pct=True)
    feats["turnover_rank_date"] = df.groupby("TS")["MEDIAN_DAILY_TURNOVER"].rank(pct=True)
    feats["turnover_x_ret1"] = feats["turnover_rank_date"] * df["RET_1"]

    # --- Cross-sectional (relative to GROUP / same-date universe) ---
    group_mean_ret1 = df.groupby(["TS", "GROUP"])["RET_1"].transform("mean")
    date_mean_ret1 = df.groupby("TS")["RET_1"].transform("mean")
    feats["ret1_minus_group_mean"] = df["RET_1"] - group_mean_ret1
    feats["ret1_minus_date_mean"] = df["RET_1"] - date_mean_ret1
    feats["ret1_pct_rank_date"] = df.groupby("TS")["RET_1"].rank(pct=True)

    feat_df = pd.DataFrame(feats, index=df.index)

    # relative 5-day performance vs group (needs avg_perf_5 already in feat_df)
    group_mean_avgperf5 = df.assign(_tmp=feat_df["avg_perf_5"]).groupby(["TS", "GROUP"])["_tmp"].transform("mean")
    feat_df["avg_perf5_minus_group_mean"] = feat_df["avg_perf_5"] - group_mean_avgperf5

    # Raw passthrough features (trees can exploit non-linear patterns directly)
    for c in RET_COLS + VOL_COLS:
        feat_df[c] = df[c]
    feat_df["MEDIAN_DAILY_TURNOVER"] = df["MEDIAN_DAILY_TURNOVER"]
    feat_df["GROUP"] = df["GROUP"].astype("category")
    feat_df["ALLOCATION"] = df["ALLOCATION"].astype("category")

    return feat_df


def get_categorical_features():
    return ["GROUP", "ALLOCATION"]


def get_linear_feature_columns(feat_df):
    """Numeric-only columns suitable for a linear model (drop high-cardinality
    ALLOCATION; GROUP is one-hot encoded separately by the caller)."""
    exclude = {"GROUP", "ALLOCATION"}
    return [c for c in feat_df.columns if c not in exclude]
