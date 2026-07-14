import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_model_zoo():
    """Return a dict of {name: sklearn-compatible estimator}."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000), cv=3),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Passive Aggressive": CalibratedClassifierCV(
            PassiveAggressiveClassifier(max_iter=1000, random_state=42), cv=3
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss", n_jobs=-1
        )
    return models


def train_and_compare(models: dict, X_train, y_train, X_test, y_test, cv_folds=5):
    """Train every model, evaluate on the test set, and return a results DataFrame + fitted models."""
    results = []
    fitted_models = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

        row = {
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, pos_label=1),
            "Recall": recall_score(y_test, preds, pos_label=1),
            "F1 Score": f1_score(y_test, preds, pos_label=1),
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
            "Train Time (s)": round(train_time, 2),
        }
        if probs is not None:
            row["ROC-AUC"] = roc_auc_score(y_test, probs)

        results.append(row)
        fitted_models[name] = model
        print(f"  ✓ {name}: Test Acc={row['Test Accuracy']:.4f}, F1={row['F1 Score']:.4f}, "
              f"CV Acc={row['CV Accuracy (mean)']:.4f} (+/-{row['CV Accuracy (std)']:.4f})")

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return results_df, fitted_models


def tune_best_model(model_name, X_train, y_train):
    """Run GridSearchCV for the best model found in comparison, if a grid is defined."""
    param_grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 5.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 20, 40],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    }

    if model_name not in param_grids:
        print(f"  No tuning grid defined for {model_name}; skipping GridSearchCV.")
        return None

    base_estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        base_estimators["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)

    estimator = base_estimators.get(model_name)
    if estimator is None:
        return None

    grid = GridSearchCV(
        estimator, param_grids[model_name], cv=3, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"  Best params for {model_name}: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_
