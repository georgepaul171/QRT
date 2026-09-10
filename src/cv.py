"""Leakage-safe cross-validation for the asset allocation forecasting challenge.

The test set is a contiguous block of dates strictly after every training
date, with the full set of allocations recurring in both. The correct CV axis
is therefore time (TS), not allocation: GroupKFold on ALLOCATION would hold
out allocations the model never actually needs to generalize to at test time.

Two splitters are provided:
  - walk_forward_splits: expanding-window, chronological train/validation
    blocks with a purge gap -- mirrors the real train-on-past/predict-on-
    future deployment setting. This is the primary CV scheme.
  - shuffled_date_kfold_splits: the benchmark's plain KFold-on-dates scheme,
    kept only as a diagnostic to show how much it overestimates accuracy by
    leaking future dates into training folds that predict past ones.
"""
import numpy as np
from sklearn.model_selection import KFold


def walk_forward_splits(ts_int_values, n_splits=6, purge=5):
    ts = np.asarray(ts_int_values)
    unique_dates = np.sort(np.unique(ts))
    n_dates = len(unique_dates)
    fold_edges = np.linspace(0, n_dates, n_splits + 1, dtype=int)

    splits = []
    for k in range(1, n_splits):
        purge_start = max(fold_edges[k] - purge, 0)
        train_dates = unique_dates[:purge_start]
        val_dates = unique_dates[fold_edges[k]:fold_edges[k + 1]]
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue
        train_idx = np.where(np.isin(ts, train_dates))[0]
        val_idx = np.where(np.isin(ts, val_dates))[0]
        splits.append((train_idx, val_idx))
    return splits


def shuffled_date_kfold_splits(ts_int_values, n_splits=8, seed=0):
    ts = np.asarray(ts_int_values)
    unique_dates = np.unique(ts)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for train_date_idx, val_date_idx in kf.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]
        train_idx = np.where(np.isin(ts, train_dates))[0]
        val_idx = np.where(np.isin(ts, val_dates))[0]
        splits.append((train_idx, val_idx))
    return splits
