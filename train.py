"""
train.py — Master training script for the Predictive Maintenance ML Pipeline.
Run this script to execute the full pipeline:
    1. Load raw data
    2. Preprocess and split
    3. Train and cross-validate all base models
    4. Tune XGBoost with GridSearchCV
    5. Train final model on full training set
    6. Evaluate final model on test set
    7. Save model, scaler, and feature list to models/
"""

from src.data_loader import load_raw_data
from src.preprocessor import split_and_preprocess, save_scaler_and_features
from src.trainer import train_and_evaluate_all, tune_xgboost, train_final_model, save_model
from src.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, compare_models


def main():
    print("\n" + "="*60)
    print(" PREDICTIVE MAINTENANCE ML PIPELINE — TRAINING")
    print("="*60)

    # Step 1: Load raw data
    print("\n[1/7] Loading data...")
    df = load_raw_data()

    # Step 2: Preprocess and split
    print("\n[2/7] Preprocessing and splitting data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_preprocess(df)

    # Step 3: Train and cross-validate all base models
    print("\n[3/7] Training and cross-validating all base models...")
    cv_results = train_and_evaluate_all(X_train, y_train)

    # Step 4: Tune XGBoost
    print("\n[4/7] Tuning XGBoost with GridSearchCV...")
    best_estimator, best_params, best_cv_f1 = tune_xgboost(X_train, y_train)

    # Step 5: Train final model on full training set
    print("\n[5/7] Training final model...")
    final_model = train_final_model(X_train, y_train, best_params)

    # Step 6: Evaluate all models on test set and compare
    print("\n[6/7] Evaluating models on test set...")
    all_metrics = []

    for name, result in cv_results.items():
        result["model"].fit(X_train, y_train)
        metrics = evaluate_model(result["model"], X_test, y_test, model_name=name)
        all_metrics.append(metrics)
        plot_confusion_matrix(result["model"], X_test, y_test, model_name=name)
        plot_roc_curve(result["model"], X_test, y_test, model_name=name)

    # Evaluate tuned XGBoost separately
    tuned_metrics = evaluate_model(final_model, X_test, y_test, model_name="XGBoost Tuned")
    all_metrics.append(tuned_metrics)
    plot_confusion_matrix(final_model, X_test, y_test, model_name="XGBoost Tuned")
    plot_roc_curve(final_model, X_test, y_test, model_name="XGBoost Tuned")
    plot_feature_importance(final_model, feature_names, model_name="XGBoost Tuned")

    # Print comparison table
    print("\nModel Comparison:")
    compare_models(all_metrics)

    # Step 7: Save artifacts
    print("\n[7/7] Saving model artifacts...")
    save_model(final_model)
    save_scaler_and_features(scaler, feature_names)

    print("\n" + "="*60)
    print(" TRAINING COMPLETE")
    print("="*60)
    print("Artifacts saved to models/")
    print("Plots saved to notebooks/plots/")


if __name__ == "__main__":
    main()
    