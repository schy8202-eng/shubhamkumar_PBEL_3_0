import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


def plot_confusion_matrix(y_test, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_model_comparison(results_df, out_dir):
    plt.figure(figsize=(9, 5))
    order = results_df.sort_values("Test Accuracy", ascending=True)
    plt.barh(order["Model"], order["Test Accuracy"], color="steelblue")
    plt.xlabel("Test Accuracy")
    plt.title("Model Comparison — Test Accuracy")
    plt.xlim(0, 1)
    for i, v in enumerate(order["Test Accuracy"]):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curves(fitted_models, X_test, y_test, out_dir):
    plt.figure(figsize=(7, 6))
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def top_predictive_words(model, feature_names, top_n=15):
    """Return top words pushing toward FAKE and toward REAL, if model exposes coefficients."""
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], importances[i]) for i in top_idx], None
    else:
        return None, None

    top_fake_idx = np.argsort(coefs)[:top_n]
    top_real_idx = np.argsort(coefs)[-top_n:][::-1]
    top_fake = [(feature_names[i], coefs[i]) for i in top_fake_idx]
    top_real = [(feature_names[i], coefs[i]) for i in top_real_idx]
    return top_fake, top_real


def full_report(y_test, y_pred, model_name):
    print(f"\n{'='*60}\nClassification Report — {model_name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
