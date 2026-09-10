"""Refit the ensemble on the full training set and generate submission.csv.

The ensemble (LightGBM + XGBoost + logistic regression, averaged) was the
consistent top performer under walk-forward CV in scripts/train.py, so it's
used here for the final submission.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw
from src.features import build_features, get_categorical_features, get_linear_feature_columns, align_categoricals
from src.models import fit_lgbm, predict_lgbm, fit_xgb, predict_xgb, fit_logreg, predict_logreg, ensemble_average

SUBMISSIONS = Path("submissions")
SUBMISSIONS.mkdir(exist_ok=True)


def main():
    X_train_raw, y_train, X_test_raw, sample_submission = load_raw("data")

    print("Building features...")
    feat_train = build_features(X_train_raw)
    feat_test = build_features(X_test_raw)
    feat_train, feat_test = align_categoricals(feat_train, feat_test)

    y_bin = (y_train > 0).astype(int)
    cat_features = get_categorical_features()

    print("Fitting LightGBM on full training set...")
    lgbm_model = fit_lgbm(feat_train, y_bin, categorical_features=cat_features, num_boost_round=800)
    lgbm_pred = predict_lgbm(lgbm_model, feat_test)

    print("Fitting XGBoost on full training set...")
    xgb_model = fit_xgb(feat_train, y_bin, n_estimators=800)
    xgb_pred = predict_xgb(xgb_model, feat_test)

    print("Fitting logistic regression on full training set...")
    lin_cols = get_linear_feature_columns(feat_train)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    grp_tr = ohe.fit_transform(feat_train[["GROUP"]])
    grp_te = ohe.transform(feat_test[["GROUP"]])
    X_tr_lin = np.hstack([feat_train[lin_cols].to_numpy(dtype=float), grp_tr])
    X_te_lin = np.hstack([feat_test[lin_cols].to_numpy(dtype=float), grp_te])
    logreg_model = fit_logreg(X_tr_lin, y_bin)
    logreg_pred = predict_logreg(logreg_model, X_te_lin)

    ens_pred = ensemble_average(lgbm_pred, xgb_pred, logreg_pred)
    predictions = (ens_pred > 0.5).astype(int)

    submission = pd.DataFrame({"prediction": predictions}, index=feat_test.index)
    submission.index.name = "ROW_ID"
    assert list(submission.index) == list(sample_submission.index), "ROW_ID order mismatch with sample_submission"

    out_path = SUBMISSIONS / "submission.csv"
    submission.to_csv(out_path)
    print(f"\nWrote {out_path} ({len(submission)} rows)")
    print(f"Predicted positive rate: {predictions.mean():.4f}")
    print(submission.head())


if __name__ == "__main__":
    main()
