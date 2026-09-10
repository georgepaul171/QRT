"""Follow-up EDA: lag-correlation profile + missingness diagnostics."""
import numpy as np
import pandas as pd

DATA = "data"
RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def main():
    X_train = pd.read_csv(f"{DATA}/X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(f"{DATA}/y_train.csv", index_col="ROW_ID")
    X_test = pd.read_csv(f"{DATA}/X_test.csv", index_col="ROW_ID")
    merged = X_train.join(y_train)

    print("=" * 70)
    print("FULL LAG CORRELATION PROFILE: RET_i vs target")
    print("=" * 70)
    for c in RET_COLS:
        print(f"{c}: corr={merged[c].corr(merged['target']):.5f}  "
              f"sign_corr={( (merged[c]>0).astype(int) ).corr((merged['target']>0).astype(int)):.5f}")

    print("\n" + "=" * 70)
    print("FULL LAG CORRELATION PROFILE: SIGNED_VOLUME_i vs target")
    print("=" * 70)
    for c in VOL_COLS:
        print(f"{c}: corr={merged[c].corr(merged['target']):.5f}  na_rate={merged[c].isna().mean():.4f}")

    print("\n" + "=" * 70)
    print("SIGNED_VOLUME_1 MISSINGNESS: train vs test, by GROUP / ALLOCATION / TS")
    print("=" * 70)
    print("train overall na rate SIGNED_VOLUME_1:", X_train["SIGNED_VOLUME_1"].isna().mean())
    print("test overall na rate SIGNED_VOLUME_1:", X_test["SIGNED_VOLUME_1"].isna().mean())

    print("\nnа rate by GROUP (train):")
    print(X_train.groupby("GROUP")["SIGNED_VOLUME_1"].apply(lambda s: s.isna().mean()))
    print("\nna rate by GROUP (test):")
    print(X_test.groupby("GROUP")["SIGNED_VOLUME_1"].apply(lambda s: s.isna().mean()))

    # is missingness concentrated in certain TS (e.g. most recent dates within train, or all of test)?
    X_train_sorted = X_train.copy()
    X_train_sorted["TS_INT"] = X_train_sorted["TS"].str.replace("DATE_", "", regex=False).astype(int)
    na_by_ts = X_train_sorted.groupby("TS_INT")["SIGNED_VOLUME_1"].apply(lambda s: s.isna().mean())
    print("\nna rate by TS_INT (train) - last 20 dates:")
    print(na_by_ts.sort_index().tail(20))
    print("\nna rate by TS_INT (train) - first 20 dates:")
    print(na_by_ts.sort_index().head(20))
    print("\noverall na rate correlation with TS_INT (train):", na_by_ts.reset_index().corr().iloc[0, 1])

    X_test_sorted = X_test.copy()
    X_test_sorted["TS_INT"] = X_test_sorted["TS"].str.replace("DATE_", "", regex=False).astype(int)
    na_by_ts_test = X_test_sorted.groupby("TS_INT")["SIGNED_VOLUME_1"].apply(lambda s: s.isna().mean())
    print("\nna rate by TS_INT (test) - first 20 dates:")
    print(na_by_ts_test.sort_index().head(20))
    print("\nna rate by TS_INT (test) - last 20 dates:")
    print(na_by_ts_test.sort_index().tail(20))

    print("\n" + "=" * 70)
    print("SKEW / KURTOSIS OF RET WINDOW vs TARGET (quick signal check)")
    print("=" * 70)
    ret_skew = X_train[RET_COLS].skew(axis=1)
    ret_kurt = X_train[RET_COLS].kurtosis(axis=1)
    print("corr(skew(RET window), target):", ret_skew.corr(y_train["target"]))
    print("corr(kurtosis(RET window), target):", ret_kurt.corr(y_train["target"]))

    print("\n" + "=" * 70)
    print("MOMENTUM/MEAN-REVERSION: RET_1 vs mean(RET_2..RET_20)")
    print("=" * 70)
    recent = X_train["RET_1"]
    older_mean = X_train[[f"RET_{i}" for i in range(2, 21)]].mean(axis=1)
    diff = recent - older_mean
    print("corr(RET_1 - mean(RET_2..20), target):", diff.corr(y_train["target"]))
    print("corr(RET_1, mean(RET_2..20)) [autocorr check]:", recent.corr(older_mean))

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL: allocation RET_1 vs same-day GROUP mean RET_1")
    print("=" * 70)
    grp_mean = X_train.groupby(["TS", "GROUP"])["RET_1"].transform("mean")
    rel = X_train["RET_1"] - grp_mean
    print("corr(RET_1 - group_mean_RET_1, target):", rel.corr(y_train["target"]))

    date_mean = X_train.groupby("TS")["RET_1"].transform("mean")
    rel_date = X_train["RET_1"] - date_mean
    print("corr(RET_1 - date_mean_RET_1, target):", rel_date.corr(y_train["target"]))


if __name__ == "__main__":
    main()
