"""Data loading for the asset allocation forecasting challenge."""
from pathlib import Path

import pandas as pd

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def load_raw(data_dir="data"):
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(data_dir / "y_train.csv", index_col="ROW_ID")["target"]
    X_test = pd.read_csv(data_dir / "X_test.csv", index_col="ROW_ID")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv", index_col="ROW_ID")
    return X_train, y_train, X_test, sample_submission


def add_ts_int(df):
    """TS is "DATE_n"; TS_INT recovers the chronological ordering as an integer."""
    df = df.copy()
    df["TS_INT"] = df["TS"].str.replace("DATE_", "", regex=False).astype(int)
    return df
