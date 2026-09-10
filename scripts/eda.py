"""One-off EDA script for the asset allocation forecasting challenge.

Answers the questions needed to design leakage-safe CV and a feature set:
  - TS continuity/shuffling per allocation
  - class balance of sign(TARGET)
  - RET / SIGNED_VOLUME distributional properties
  - behavior across GROUP
  - overlap of ALLOCATION and TS between train and test
  - missingness
"""
import numpy as np
import pandas as pd

DATA = "data"

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def main():
    X_train = pd.read_csv(f"{DATA}/X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(f"{DATA}/y_train.csv", index_col="ROW_ID")
    X_test = pd.read_csv(f"{DATA}/X_test.csv", index_col="ROW_ID")

    print("=" * 70)
    print("SHAPES")
    print("=" * 70)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("MISSINGNESS")
    print("=" * 70)
    na_train = X_train.isna().mean().sort_values(ascending=False)
    print(na_train[na_train > 0])
    na_test = X_test.isna().mean().sort_values(ascending=False)
    print("test:")
    print(na_test[na_test > 0])

    print("\n" + "=" * 70)
    print("TS / ALLOCATION STRUCTURE")
    print("=" * 70)
    print(f"train unique TS: {X_train['TS'].nunique()}, unique ALLOCATION: {X_train['ALLOCATION'].nunique()}")
    print(f"test unique TS: {X_test['TS'].nunique()}, unique ALLOCATION: {X_test['ALLOCATION'].nunique()}")
    print(f"rows per TS (train) describe:\n{X_train.groupby('TS').size().describe()}")
    print(f"rows per ALLOCATION (train) describe:\n{X_train.groupby('ALLOCATION').size().describe()}")

    train_ts = set(X_train["TS"].unique())
    test_ts = set(X_test["TS"].unique())
    print(f"\nTS overlap train/test: {len(train_ts & test_ts)} shared out of {len(test_ts)} test TS values")
    print(f"test TS values: {sorted(test_ts)[:10]}{'...' if len(test_ts) > 10 else ''}")

    train_alloc = set(X_train["ALLOCATION"].unique())
    test_alloc = set(X_test["ALLOCATION"].unique())
    print(f"\nALLOCATION overlap train/test: {len(train_alloc & test_alloc)} shared "
          f"out of {len(test_alloc)} test / {len(train_alloc)} train")

    # Does TS look like it encodes a real chronological order, or is it shuffled?
    # Check: for a given allocation, do consecutive TS_int labels (if TS is "DATE_n") correspond
    # to actually adjacent RET windows (i.e. RET_1 at TS=n should relate to RET_20 at TS=n+19)?
    print("\n" + "=" * 70)
    print("TS ORDERING CHECK (does DATE_n order match chronology within an allocation?)")
    print("=" * 70)
    X_train_sorted = X_train.copy()
    X_train_sorted["TS_INT"] = X_train_sorted["TS"].str.replace("DATE_", "", regex=False).astype(int)
    sample_alloc = X_train_sorted["ALLOCATION"].value_counts().index[0]
    sub = X_train_sorted[X_train_sorted["ALLOCATION"] == sample_alloc].sort_values("TS_INT")
    print(f"Sample allocation: {sample_alloc}, n_rows={len(sub)}")
    print(sub[["TS_INT", "RET_1", "RET_2", "RET_3"]].head(10))
    # gaps in TS_INT for this allocation
    gaps = sub["TS_INT"].diff().dropna()
    print(f"TS_INT gap distribution for this allocation:\n{gaps.value_counts().head(10)}")

    # Cross-check: RET_1 at TS=t should approx equal RET_2 at TS=t+1 if TS_INT is a real daily clock
    # and RET indices shift by one day. Check correlation.
    sub2 = sub.set_index("TS_INT")
    shifted = sub2["RET_2"].shift(-1)  # RET_2 at t+1
    aligned = pd.DataFrame({"RET_1_t": sub2["RET_1"], "RET_2_t+1": shifted}).dropna()
    if len(aligned) > 5:
        print(f"corr(RET_1[t], RET_2[t+1]) for consecutive TS_INT, n={len(aligned)}: "
              f"{aligned['RET_1_t'].corr(aligned['RET_2_t+1']):.4f}")

    print("\n" + "=" * 70)
    print("CLASS BALANCE")
    print("=" * 70)
    sign = (y_train["target"] > 0).astype(int)
    print(sign.value_counts(normalize=True))
    print(f"mean target: {y_train['target'].mean():.6f}, std: {y_train['target'].std():.6f}")

    print("\n" + "=" * 70)
    print("RET / SIGNED_VOLUME DISTRIBUTIONS")
    print("=" * 70)
    print("RET_1 describe:\n", X_train["RET_1"].describe())
    print("SIGNED_VOLUME_1 describe:\n", X_train["SIGNED_VOLUME_1"].describe())
    print(f"skew RET_1: {X_train['RET_1'].skew():.4f}, kurtosis: {X_train['RET_1'].kurtosis():.4f}")

    print("\n" + "=" * 70)
    print("GROUP BEHAVIOR")
    print("=" * 70)
    print(f"n groups: {X_train['GROUP'].nunique()}")
    print(X_train["GROUP"].value_counts())
    merged = X_train.join(y_train)
    grp_stats = merged.groupby("GROUP").agg(
        mean_target=("target", "mean"),
        pct_positive=("target", lambda s: (s > 0).mean()),
        n=("target", "size"),
    )
    print(grp_stats)

    print("\n" + "=" * 70)
    print("TARGET CORRELATION WITH SIMPLE FEATURES (sanity)")
    print("=" * 70)
    for c in ["RET_1", "RET_2", "RET_3", "SIGNED_VOLUME_1", "MEDIAN_DAILY_TURNOVER"]:
        print(f"corr(target, {c}) = {merged['target'].corr(merged[c]):.5f}")


if __name__ == "__main__":
    main()
