
# ============================================================
# STAGE 5.3: FINAL EVALUATION OF TUNED MODEL
# Project: Aspect-Based Sentiment Analysis
# ============================================================

import os
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


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = (
    r"C:\Users\manoj\OneDrive\Desktop\Aspect-Based Sentiment Analysis"
)

DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

DATA_FILE = os.path.join(DATA_FOLDER, "Laptop_Train_v2.csv")

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 70)
print("STAGE 5.3: FINAL EVALUATION OF TUNED MODEL")
print("=" * 70)


# ------------------------------------------------------------
# 3. LOAD ORIGINAL DATASET
# ------------------------------------------------------------

print("\nLoading original dataset...")

df = pd.read_csv(DATA_FILE)

df["combined_text"] = (
    df["Sentence"].astype(str)
    + " targetaspect "
    + df["Aspect Term"].astype(str)
)

X = df["combined_text"]
y = df["polarity"].astype(str)
groups = df["id"]


# ------------------------------------------------------------
# 4. RECREATE ORIGINAL STAGE 3 SPLIT
# ------------------------------------------------------------

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_test_text = X.iloc[test_indices].reset_index(drop=True)
y_test_text = y.iloc[test_indices].reset_index(drop=True)


# ------------------------------------------------------------
# 5. LOAD SAVED TEST DATA
# ------------------------------------------------------------

print("\nLoading original Stage 3 test files...")

X_test_tfidf = load_npz(
    os.path.join(MODELS_FOLDER, "X_test_tfidf.npz")
)

y_test_saved = np.load(
    os.path.join(MODELS_FOLDER, "y_test.npy"),
    allow_pickle=False
)

if len(X_test_text) != 478:
    raise ValueError("Unexpected test record count.")

if not np.array_equal(
    y_test_text.to_numpy(dtype=str),
    y_test_saved.astype(str)
):
    raise ValueError("Test labels do not match Stage 3.")

if X_test_tfidf.shape[0] != len(y_test_saved):
    raise ValueError("Test matrix and labels have different sizes.")

print("Original test set verified.")
print("Test records:", len(y_test_saved))


# ------------------------------------------------------------
# 6. LOAD THREE TRAINED MODELS
# ------------------------------------------------------------

print("\nLoading trained models...")

logistic_model = joblib.load(
    os.path.join(
        MODELS_FOLDER,
        "logistic_regression_model.joblib"
    )
)

baseline_svm_model = joblib.load(
    os.path.join(
        MODELS_FOLDER,
        "linear_svm_model.joblib"
    )
)

tuned_svm_model = joblib.load(
    os.path.join(
        MODELS_FOLDER,
        "tuned_linear_svm_model.joblib"
    )
)

print("All three models loaded successfully.")


# ------------------------------------------------------------
# 7. GENERATE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

# Baseline models use the saved Stage 3 TF-IDF matrix.
logistic_predictions = logistic_model.predict(X_test_tfidf)

baseline_svm_predictions = baseline_svm_model.predict(
    X_test_tfidf
)

# Tuned model contains its own TF-IDF vectorizer.
tuned_svm_predictions = tuned_svm_model.predict(
    X_test_text
)

y_true = y_test_saved.astype(str)

print("Predictions generated successfully.")


# ------------------------------------------------------------
# 8. EVALUATION FUNCTION
# ------------------------------------------------------------

class_labels = [
    "conflict",
    "negative",
    "neutral",
    "positive"
]


def evaluate_model(model_name, predictions):

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Macro Precision": precision_score(
            y_true,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y_true,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "Weighted F1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return metrics


# ------------------------------------------------------------
# 9. CALCULATE METRICS
# ------------------------------------------------------------

print("\nCalculating evaluation metrics...")

all_predictions = {
    "Logistic Regression": logistic_predictions,
    "Baseline Linear SVM": baseline_svm_predictions,
    "Tuned Linear SVM": tuned_svm_predictions
}

comparison_results = []

for model_name, predictions in all_predictions.items():

    comparison_results.append(
        evaluate_model(model_name, predictions)
    )

comparison_df = pd.DataFrame(comparison_results)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# 10. CLASSIFICATION REPORTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)

report_text = []

for model_name, predictions in all_predictions.items():

    report = classification_report(
        y_true,
        predictions,
        labels=class_labels,
        zero_division=0
    )

    print("\n", model_name)
    print("-" * 60)
    print(report)

    report_text.append(
        "\n" + model_name + "\n"
        + "-" * 60 + "\n"
        + report
    )


# ------------------------------------------------------------
# 11. CONFUSION MATRICES
# ------------------------------------------------------------

print("\nGenerating confusion matrices...")

for model_name, predictions in all_predictions.items():

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=class_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        model_name + " - Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + "_final_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(RESULTS_FOLDER, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ------------------------------------------------------------
# 12. MACRO-F1 COMPARISON CHART
# ------------------------------------------------------------

print("\nCreating Macro-F1 comparison chart...")

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    comparison_df["Model"],
    comparison_df["Macro F1"]
)

ax.set_title("Macro-F1 Comparison of Sentiment Models")
ax.set_ylabel("Macro-F1 Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)

ax.tick_params(
    axis="x",
    rotation=15
)

for index, value in enumerate(comparison_df["Macro F1"]):

    ax.text(
        index,
        value + 0.015,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "stage5_3_final_macro_f1_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 13. IDENTIFY BEST TEST MODEL
# ------------------------------------------------------------

best_row = comparison_df.loc[
    comparison_df["Macro F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL ON HELD-OUT TEST SET")
print("=" * 70)

print("Model:", best_row["Model"])
print("Accuracy:", round(best_row["Accuracy"], 4))
print("Macro-F1:", round(best_row["Macro F1"], 4))


# ------------------------------------------------------------
# 14. SAVE FINAL EVALUATION RESULTS
# ------------------------------------------------------------

summary_path = os.path.join(
    RESULTS_FOLDER,
    "stage5_3_final_evaluation.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 5.3: FINAL MODEL EVALUATION\n")
    file.write("=" * 60 + "\n\n")

    file.write("Test records: 478\n")
    file.write("Evaluation data: Original Stage 3 held-out test set\n")
    file.write("Models compared: 3\n\n")

    file.write("FINAL METRICS:\n\n")
    file.write(
        comparison_df.round(4).to_string(index=False)
    )

    file.write("\n\nCLASSIFICATION REPORTS:\n")

    for report in report_text:
        file.write(report + "\n")

    file.write("\nBEST MODEL ON TEST SET:\n")
    file.write(f"Model: {best_row['Model']}\n")
    file.write(
        f"Accuracy: {best_row['Accuracy']:.4f}\n"
    )
    file.write(
        f"Macro-F1: {best_row['Macro F1']:.4f}\n"
    )


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\nSaved files:")
print("1. results/stage5_3_final_evaluation.txt")
print("2. results/stage5_3_final_macro_f1_comparison.png")
print("3. Three final confusion matrix images")

print("\n" + "=" * 70)
print("STAGE 5.3 COMPLETED SUCCESSFULLY")
print("=" * 70)
