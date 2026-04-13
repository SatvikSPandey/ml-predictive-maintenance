import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from xgboost import XGBClassifier


# Cross-validation strategy: 5 folds, stratified to preserve failure rate in each fold
CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics to collect during cross-validation
SCORING = {
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score),
}


def get_base_models() -> dict:
    """
    Returns a dictionary of model name -> untrained model instance.
    scale_pos_weight in XGBoost handles class imbalance by telling the model
    to penalise missing a failure more than missing a non-failure.
    ~96.6% non-failure / ~3.4% failure = ratio of about 28.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=28,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def train_and_evaluate_all(X_train, y_train) -> dict:
    """
    Trains all base models using 5-fold stratified cross-validation.
    Returns a dictionary of results with mean F1 and ROC-AUC for each model.
    """
    models = get_base_models()
    results = {}

    for name, model in models.items():
        print(f"Training: {name}...")
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=CV_STRATEGY,
            scoring=SCORING,
            return_train_score=False,
        )
        mean_f1 = np.mean(cv_results["test_f1"])
        mean_auc = np.mean(cv_results["test_roc_auc"])
        print(f"  F1: {mean_f1:.4f} | ROC-AUC: {mean_auc:.4f}")
        results[name] = {
            "model": model,
            "mean_f1": mean_f1,
            "mean_auc": mean_auc,
        }

    return results


def tune_xgboost(X_train, y_train) -> tuple:
    """
    Runs GridSearchCV to find the best XGBoost hyperparameters.
    GridSearchCV tries every combination of the parameter grid and
    returns the combination that gives the best cross-validated F1 score.
    """
    print("Tuning XGBoost with GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    xgb = XGBClassifier(
        scale_pos_weight=28,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1",
        cv=CV_STRATEGY,
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_final_model(X_train, y_train, best_params: dict):
    """
    Trains the final XGBoost model with the best hyperparameters
    on the full training set (not just one fold).
    This is the model we save and deploy.
    """
    print("Training final model on full training set...")
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=28,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    final_model.fit(X_train, y_train)
    print("Final model trained.")
    return final_model


def save_model(model, filepath: str = "models/best_model.pkl"):
    """
    Serializes the trained model to disk using joblib.
    joblib is preferred over pickle for scikit-learn and XGBoost models
    because it handles large numpy arrays more efficiently.
    """
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")