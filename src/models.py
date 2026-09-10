"""Model wrappers: LightGBM / XGBoost classifiers, logistic regression baseline,
and a simple probability-averaging ensemble."""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    num_threads=8,
    seed=42,
    verbosity=-1,
    learning_rate=0.02,
    max_depth=4,
    num_leaves=15,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=200,
)

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    tree_method="hist",
    enable_categorical=True,
    n_estimators=800,
    n_jobs=8,
    random_state=42,
)


def fit_lgbm(X_train, y_train, X_val=None, y_val=None, categorical_features=None, num_boost_round=800):
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = []
    if X_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_set)
        valid_sets.append(val_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))
    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def predict_lgbm(model, X):
    best_iter = getattr(model, "best_iteration", None)
    return model.predict(X, num_iteration=best_iter)


def fit_xgb(X_train, y_train, X_val=None, y_val=None, n_estimators=800):
    params = dict(XGB_PARAMS)
    params["n_estimators"] = n_estimators
    model = xgb.XGBClassifier(**params)
    fit_kwargs = {}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_xgb(model, X):
    return model.predict_proba(X)[:, 1]


def fit_logreg(X_train, y_train):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def predict_logreg(model, X):
    return model.predict_proba(X)[:, 1]


def ensemble_average(*prob_arrays, weights=None):
    arrs = np.stack(prob_arrays, axis=0)
    if weights is None:
        return arrs.mean(axis=0)
    w = np.asarray(weights).reshape(-1, 1)
    return (arrs * w).sum(axis=0) / w.sum()
