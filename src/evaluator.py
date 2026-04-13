import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
)


# Directory where all plots will be saved
PLOTS_DIR = Path("notebooks/plots")


def ensure_plots_dir():
    """Creates the plots directory if it doesn't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Runs full evaluation of a trained model against the test set.
    Returns a dictionary of all computed metrics.
    """
    # Get predictions (0 or 1) and probabilities (0.0 to 1.0)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of failure (class 1)

    # Compute all metrics
    metrics = {
        "model_name": model_name,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*50}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model"):
    """
    Plots and saves a confusion matrix heatmap.
    Shows exactly how many failures were caught, missed, and falsely flagged.
    """
    ensure_plots_dir()

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    filename = PLOTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def plot_roc_curve(model, X_test, y_test, model_name: str = "Model"):
    """
    Plots and saves the ROC curve.
    The ROC curve shows the tradeoff between catching failures (recall)
    and avoiding false alarms (1 - specificity) at different thresholds.
    """
    ensure_plots_dir()

    y_prob = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title(f"ROC Curve — {model_name}")
    plt.tight_layout()

    filename = PLOTS_DIR / f"roc_curve_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"ROC curve saved to {filename}")


def plot_feature_importance(model, feature_names: list, model_name: str = "XGBoost"):
    """
    Plots and saves feature importances from the trained XGBoost model.
    Shows which sensor readings matter most for predicting failure.
    """
    ensure_plots_dir()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    filename = PLOTS_DIR / f"feature_importance_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {filename}")


def compare_models(results: list) -> None:
    """
    Prints a comparison table of all models by F1, Precision, Recall, and ROC-AUC.
    results: list of metric dictionaries returned by evaluate_model()
    """
    print(f"\n{'='*65}")
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}")
    print(f"{'='*65}")
    for r in results:
        print(
            f"{r['model_name']:<25} {r['f1']:>8.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['roc_auc']:>8.4f}"
        )
    print(f"{'='*65}")