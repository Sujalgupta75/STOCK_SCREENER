from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from .features import FEATURES

def _pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", RobustScaler(quantile_range=(10, 90))))
    steps.append(("model", model))
    return Pipeline(steps)

def chronological_split(df: pd.DataFrame, test_fraction=0.20, val_fraction=0.20, purge=5):
    """Split by unique dates, leaving a purge gap between periods for forward labels."""
    dates = np.array(sorted(pd.to_datetime(df["Date"]).dropna().unique()))
    if len(dates) < 80:
        raise ValueError(f"Need at least 80 distinct dates; found {len(dates)}. Choose more history.")
    test_start_i = int(len(dates) * (1 - test_fraction))
    val_start_i = int(len(dates) * (1 - test_fraction - val_fraction))
    # Purge around boundaries because the target uses future prices.
    train_end_i = max(1, val_start_i - purge)
    val_start_i = min(val_start_i + purge, test_start_i - 1)
    val_end_i = max(val_start_i + 1, test_start_i - purge)
    test_start_i = min(test_start_i + purge, len(dates) - 1)
    train_dates = dates[:train_end_i]
    val_dates = dates[val_start_i:val_end_i]
    test_dates = dates[test_start_i:]
    date_series = pd.to_datetime(df["Date"])
    return (
        date_series.isin(train_dates).to_numpy(),
        date_series.isin(val_dates).to_numpy(),
        date_series.isin(test_dates).to_numpy(),
    )

def _safe_metrics(y_true, pred, prob):
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    prob = np.asarray(prob, dtype=float)
    valid_prob = np.isfinite(prob)
    auc = float(roc_auc_score(y_true[valid_prob], prob[valid_prob])) if len(np.unique(y_true[valid_prob])) == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, pred, labels=[0, 1], target_names=["Non-positive", "Positive"], zero_division=0, output_dict=True),
    }

def _model_specs():
    return {
        "Logistic Regression": (
            _pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42), scale=True),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        "Random Forest": (
            _pipeline(RandomForestClassifier(n_estimators=180, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [3, 8], "model__max_features": ["sqrt", 0.8]}
        ),
        "Hist Gradient Boosting": (
            _pipeline(HistGradientBoostingClassifier(max_iter=120, early_stopping=True, random_state=42)),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.0, 1.0]}
        ),
    }

def train_and_evaluate(featured: pd.DataFrame, tune=True):
    usable = featured.dropna(subset=["Target"]).copy()
    usable = usable[np.isfinite(usable["Target"])]
    usable["Target"] = usable["Target"].astype(int)
    # Require all feature columns to be represented; imputer handles partial missingness.
    usable = usable.dropna(subset=["Date", "Ticker"])
    if len(usable) < 250:
        raise ValueError(f"Only {len(usable)} labelled rows remain. Use more tickers or a longer date range.")
    train_mask, val_mask, test_mask = chronological_split(usable)
    train = usable.loc[train_mask]
    val = usable.loc[val_mask]
    test = usable.loc[test_mask]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split produced an empty period. Increase the date range.")
    if train["Target"].nunique() < 2:
        raise ValueError("Training data contains only one target class. Use more data or a different period.")
    if val["Target"].nunique() < 2:
        warnings.warn("Validation period contains one class; ROC-AUC cannot be used reliably.")
    X_train, y_train = train[FEATURES], train["Target"]
    X_val, y_val = val[FEATURES], val["Target"]
    X_test, y_test = test[FEATURES], test["Target"]
    # TimeSeriesSplit acts on chronological unique dates; rows from same date should not cross folds.
    unique_train_dates = np.array(sorted(train["Date"].unique()))
    n_splits = min(3, max(2, len(unique_train_dates) // 80))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=5)
    date_to_pos = {d: i for i, d in enumerate(unique_train_dates)}
    date_positions = train["Date"].map(date_to_pos).to_numpy()
    # GridSearchCV expects row-level indices; construct folds based on unique date positions.
    cv = []
    for tr_idx, va_idx in splitter.split(unique_train_dates):
        tr_dates = set(unique_train_dates[tr_idx])
        va_dates = set(unique_train_dates[va_idx])
        tr_rows = np.flatnonzero(train["Date"].isin(tr_dates).to_numpy())
        va_rows = np.flatnonzero(train["Date"].isin(va_dates).to_numpy())
        if len(tr_rows) and len(va_rows):
            cv.append((tr_rows, va_rows))
    if not cv:
        raise ValueError("Not enough dates for time-series cross-validation.")
    results = {}
    fitted = {}
    for name, (pipe, grid) in _model_specs().items():
        try:
            if tune:
                search = GridSearchCV(
                    pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                    refit=True, error_score=np.nan, return_train_score=False
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_score = float(search.best_score_) if np.isfinite(search.best_score_) else None
            else:
                model = clone(pipe).fit(X_train, y_train)
                best_params, cv_score = {}, None
            val_prob = model.predict_proba(X_val)[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            val_metrics = _safe_metrics(y_val, val_pred, val_prob)
            selection_score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["f1"]
            results[name] = {
                "validation": val_metrics, "cv_roc_auc": cv_score,
                "best_params": best_params, "selection_score": float(selection_score),
                "error": None
            }
            fitted[name] = model
        except Exception as exc:
            results[name] = {"validation": None, "cv_roc_auc": None, "best_params": {}, "selection_score": -1, "error": f"{type(exc).__name__}: {exc}"}
    valid = [(n, r["selection_score"]) for n, r in results.items() if r["validation"] is not None]
    if not valid:
        errors = "; ".join(f"{n}: {r['error']}" for n, r in results.items())
        raise RuntimeError("All models failed. " + errors)
    best_name = max(valid, key=lambda x: x[1])[0]
    best_model = fitted[best_name]
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = _safe_metrics(y_test, test_pred, test_prob)
    # Fit final deployment model on all labelled observations up to the test start.
    # Keep held-out test metrics from the model selected on train/validation, do not train on test for evaluation.
    final_model = best_model
    feature_importance = []
    est = final_model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        feature_importance = sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(FEATURES, est.feature_importances_)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(est, "coef_"):
        feature_importance = sorted(
            [{"feature": f, "importance": float(abs(v))} for f, v in zip(FEATURES, np.ravel(est.coef_))],
            key=lambda x: x["importance"], reverse=True
        )
    return {
        "best_name": best_name, "best_model": final_model, "models": fitted,
        "results": results, "test_metrics": test_metrics,
        "test_dates": {"start": str(pd.Timestamp(test["Date"].min()).date()), "end": str(pd.Timestamp(test["Date"].max()).date())},
        "test_rows": int(len(test)), "train_rows": int(len(train)), "validation_rows": int(len(val)),
        "feature_importance": feature_importance, "features": FEATURES,
    }

def score_latest(featured: pd.DataFrame, model, model_name="Model"):
    latest = featured.sort_values(["Ticker", "Date"]).groupby("Ticker", as_index=False).tail(1).copy()
    X = latest[FEATURES].replace([np.inf, -np.inf], np.nan)
    latest["Prediction_Probability"] = model.predict_proba(X)[:, 1]
    latest["Signal"] = np.select(
        [latest["Prediction_Probability"] >= 0.60, latest["Prediction_Probability"] <= 0.40],
        ["Positive-return candidate", "Lower-probability candidate"],
        default="Watch"
    )
    latest["Model"] = model_name
    return latest.sort_values("Prediction_Probability", ascending=False).reset_index(drop=True)
