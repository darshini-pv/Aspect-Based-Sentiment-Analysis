
# ============================================================
# ASPECT-BASED SENTIMENT ANALYSIS
# STAGE 5: MODEL EVALUATION
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_FOLDER = (
    r"C:\Users\manoj\OneDrive\Desktop\Aspect-Based Sentiment Analysis"
)

MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("=" * 65)
print("STAGE 5: MODEL EVALUATION")
print("=" * 65)

# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = load_npz(
    os.path.join(MODELS_FOLDER, "X_test_tfidf.npz")
)

y_test = np.load(
    os.path.join(MODELS_FOLDER, "y_test.npy"),
    allow_pickle=True
)

print("Test data loaded successfully.")
print("Testing feature shape:", X_test.shape)
print("Testing labels:", len(y_test))

# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

print("\nLoading trained models...")

logistic_model = joblib.load(
    os.path.join(MODELS_FOLDER, "logistic_regression_model.joblib")
)

svm_model = joblib.load(
    os.path.join(MODELS_FOLDER, "linear_svm_model.joblib")
)

print("Logistic Regression loaded.")
print("Linear SVM loaded.")

# ============================================================
# 4. VERIFY DATA AND MODELS
# ============================================================

if X_test.shape[0] != len(y_test):
    raise ValueError("Test features and labels have different sizes.")

if X_test.shape[1] != logistic_model.n_features_in_:
    raise ValueError("Logistic Regression feature mismatch.")

if X_test.shape[1] != svm_model.n_features_in_:
    raise ValueError("Linear SVM feature mismatch.")

LABELS = ["conflict", "negative", "neutral", "positive"]

print("\nData verification successful.")

print("\nActual testing sentiment distribution:")
print(pd.Series(y_test).value_counts().reindex(LABELS, fill_value=0))

# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

start_time = time.time()
logistic_predictions = logistic_model.predict(X_test)
logistic_prediction_time = time.time() - start_time

start_time = time.time()
svm_predictions = svm_model.predict(X_test)
svm_prediction_time = time.time() - start_time

print("Predictions generated successfully.")

# ============================================================
# 6. EVALUATE BOTH MODELS
# ============================================================

models = {
    "Logistic Regression": logistic_predictions,
    "Linear SVM": svm_predictions
}

comparison_results = []

for model_name, predictions in models.items():

    print("\n" + "=" * 65)
    print("EVALUATION:", model_name)
    print("=" * 65)

    accuracy = accuracy_score(y_test, predictions)

    precision_macro = precision_score(
        y_test, predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test, predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test, predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-score:     {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-score:  {f1_weighted:.4f}")

    # Classification report

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_filename = (
        "logistic_regression_classification_report.txt"
        if model_name == "Logistic Regression"
        else "linear_svm_classification_report.txt"
    )

    report_path = os.path.join(
        RESULTS_FOLDER, report_filename
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(model_name + "\n")
        file.write("=" * 60 + "\n\n")
        file.write(report)

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(model_name + " - Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    image_filename = (
        "confusion_matrix_logistic_regression.png"
        if model_name == "Logistic Regression"
        else "confusion_matrix_linear_svm.png"
    )

    plt.savefig(
        os.path.join(RESULTS_FOLDER, image_filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    comparison_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision_macro,
        "Macro Recall": recall_macro,
        "Macro F1": f1_macro,
        "Weighted Precision": precision_weighted,
        "Weighted Recall": recall_weighted,
        "Weighted F1": f1_weighted
    })

# ============================================================
# 7. MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(comparison_results)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(comparison_df.to_string(index=False))

# Save comparison as text because the current .gitignore
# excludes CSV files.

comparison_path = os.path.join(
    RESULTS_FOLDER,
    "stage5_model_comparison.txt"
)

with open(comparison_path, "w", encoding="utf-8") as file:
    file.write("STAGE 5: MODEL PERFORMANCE COMPARISON\n")
    file.write("=" * 65 + "\n\n")
    file.write(comparison_df.to_string(index=False))
    file.write("\n\nPrimary evaluation metric: Macro F1-score\n")
    file.write(
        "Note: Conflict class performance is unstable because "
        "only four conflict examples occur in the test set.\n"
    )

# ============================================================
# 8. COMPARE MACRO F1 SCORES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    comparison_df["Model"],
    comparison_df["Macro F1"]
)

ax.set_title("Macro F1-score Comparison")
ax.set_ylabel("Macro F1-score")
ax.set_ylim(0, 1)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_FOLDER, "stage5_macro_f1_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# 9. FINAL OBSERVATION
# ============================================================

best_model = comparison_df.loc[
    comparison_df["Macro F1"].idxmax()
]

print("\n" + "=" * 65)
print("STAGE 5 FINAL OBSERVATION")
print("=" * 65)

print("Best model based on Macro F1:", best_model["Model"])
print(f"Macro F1-score: {best_model['Macro F1']:.4f}")

print("\nImportant evaluation limitation:")
print(
    "Only four conflict examples are present in the test set. "
    "Therefore, conflict-class precision, recall and F1 are "
    "not statistically reliable."
)

print("\nAll evaluation files saved in:")
print(RESULTS_FOLDER)

print("\nSaved files:")
print("1. stage5_model_comparison.txt")
print("2. logistic_regression_classification_report.txt")
print("3. linear_svm_classification_report.txt")
print("4. confusion_matrix_logistic_regression.png")
print("5. confusion_matrix_linear_svm.png")
print("6. stage5_macro_f1_comparison.png")

print("\n" + "=" * 65)
print("STAGE 5 COMPLETED SUCCESSFULLY")
print("=" * 65)
