
# ============================================================
# STAGE 6: FINAL MODEL INTERPRETATION AND ERROR ANALYSIS
# Project: Aspect-Based Sentiment Analysis
#
# Selected final model: Baseline Linear SVM
# Purpose:
# 1. Analyse correct and incorrect predictions.
# 2. Calculate class-wise error statistics.
# 3. Identify common sentiment confusion patterns.
# 4. Save misclassified examples and a final report.
#
# IMPORTANT:
# This stage only analyses the existing test predictions.
# It does not retrain or tune the model.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "Laptop_Train_v2.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "linear_svm_model.joblib"
)

X_TEST_FILE = os.path.join(
    BASE_DIR, "models", "X_test_tfidf.npz"
)

Y_TEST_FILE = os.path.join(
    BASE_DIR, "models", "y_test.npy"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed sentiment class order
LABELS = ["conflict", "negative", "neutral", "positive"]


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

print("=" * 70)
print("STAGE 6: FINAL MODEL INTERPRETATION AND ERROR ANALYSIS")
print("=" * 70)

required_files = [
    DATA_FILE,
    MODEL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Required file not found:\n" + file_path
        )

print("\nAll required files are available.")


# ============================================================
# 3. LOAD ORIGINAL DATASET AND RECONSTRUCT TEST SPLIT
# ============================================================

print("\nLoading original labelled dataset...")

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)

# Reconstruct the exact Stage 3 grouped split.
# Keep these settings unchanged.

X = (
    df["Sentence"].astype(str)
    + " targetaspect "
    + df["Aspect Term"].astype(str)
)

y = df["polarity"].astype(str)
groups = df["id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

test_df = df.iloc[test_indices].copy().reset_index(drop=True)

print("Reconstructed test records:", len(test_df))


# ============================================================
# 4. LOAD SAVED TEST FEATURES AND LABELS
# ============================================================

print("\nLoading saved test features and labels...")

from scipy.sparse import load_npz

X_test = load_npz(X_TEST_FILE)
y_test = np.load(Y_TEST_FILE, allow_pickle=True)

y_test = np.asarray(y_test).astype(str)

print("Test feature shape:", X_test.shape)
print("Test label count:", len(y_test))

# Verify that the reconstructed split matches Stage 3.

if len(test_df) != len(y_test):
    raise ValueError(
        "Test dataset row count does not match saved test labels."
    )

if not np.array_equal(
    test_df["polarity"].astype(str).values,
    y_test
):
    raise ValueError(
        "Reconstructed test labels do not match saved Stage 3 labels."
    )

if X_test.shape[0] != len(y_test):
    raise ValueError(
        "Test feature rows and test labels do not match."
    )

print("Original test split verified successfully.")


# ============================================================
# 5. LOAD THE SELECTED FINAL MODEL
# ============================================================

print("\nLoading selected final model...")

model = joblib.load(MODEL_FILE)

print("Baseline Linear SVM loaded successfully.")


# ============================================================
# 6. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Predictions generated:", len(y_pred))


# ============================================================
# 7. CREATE PREDICTION ANALYSIS DATAFRAME
# ============================================================

analysis_df = test_df[
    ["id", "Sentence", "Aspect Term", "polarity"]
].copy()

analysis_df = analysis_df.rename(
    columns={"polarity": "actual_sentiment"}
)

analysis_df["predicted_sentiment"] = y_pred

analysis_df["prediction_status"] = np.where(
    analysis_df["actual_sentiment"]
    == analysis_df["predicted_sentiment"],
    "Correct",
    "Incorrect"
)

# Save all test predictions for local inspection.

all_predictions_file = os.path.join(
    RESULTS_DIR,
    "stage6_all_test_predictions.csv"
)

analysis_df.to_csv(
    all_predictions_file,
    index=False
)

correct_df = analysis_df[
    analysis_df["prediction_status"] == "Correct"
]

incorrect_df = analysis_df[
    analysis_df["prediction_status"] == "Incorrect"
]

print("\nPrediction analysis completed.")

print("Correct predictions:", len(correct_df))
print("Incorrect predictions:", len(incorrect_df))


# ============================================================
# 8. OVERALL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        zero_division=0
    )
)

class_report = classification_report(
    y_test,
    y_pred,
    labels=LABELS,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=LABELS
)


# ============================================================
# 9. CLASS-WISE ERROR ANALYSIS
# ============================================================

class_summary = []

for i, label in enumerate(LABELS):

    total = int(support[i])

    correct = int(cm[i, i])

    errors = total - correct

    error_rate = (
        errors / total * 100
        if total > 0 else 0
    )

    class_summary.append({
        "Sentiment": label,
        "Total_Records": total,
        "Correct_Predictions": correct,
        "Incorrect_Predictions": errors,
        "Recall": recall[i],
        "Error_Rate_Percent": error_rate
    })

class_summary_df = pd.DataFrame(class_summary)


# ============================================================
# 10. IDENTIFY COMMON CONFUSION PATTERNS
# ============================================================

confusion_pairs = []

for i, actual_label in enumerate(LABELS):

    for j, predicted_label in enumerate(LABELS):

        if i != j and cm[i, j] > 0:

            confusion_pairs.append({
                "Actual_Sentiment": actual_label,
                "Predicted_Sentiment": predicted_label,
                "Number_of_Records": int(cm[i, j])
            })

confusion_pairs_df = pd.DataFrame(
    confusion_pairs
)

if not confusion_pairs_df.empty:
    confusion_pairs_df = confusion_pairs_df.sort_values(
        by="Number_of_Records",
        ascending=False
    )


# ============================================================
# 11. SAVE MISCLASSIFIED EXAMPLES
# ============================================================

misclassified_file = os.path.join(
    RESULTS_DIR,
    "stage6_misclassified_examples.csv"
)

incorrect_df.to_csv(
    misclassified_file,
    index=False
)


# ============================================================
# 12. GENERATE CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(8, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Stage 6: Baseline Linear SVM Error Analysis"
)

plt.tight_layout()

confusion_plot_file = os.path.join(
    RESULTS_DIR,
    "stage6_error_analysis_confusion_matrix.png"
)

plt.savefig(
    confusion_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. GENERATE FINAL WRITTEN REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "stage6_error_analysis.txt"
)

with open(report_file, "w", encoding="utf-8") as file:

    file.write("=" * 70 + "\n")
    file.write("STAGE 6: FINAL MODEL ERROR ANALYSIS\n")
    file.write("=" * 70 + "\n\n")

    file.write("Selected Model: Baseline Linear SVM\n")
    file.write("Dataset: SemEval-2014 Laptop ABSA\n")
    file.write("Evaluation: Original held-out test set\n\n")

    file.write("-" * 70 + "\n")
    file.write("1. OVERALL PREDICTION SUMMARY\n")
    file.write("-" * 70 + "\n")

    file.write(
        f"Total test records: {len(analysis_df)}\n"
    )

    file.write(
        f"Correct predictions: {len(correct_df)}\n"
    )

    file.write(
        f"Incorrect predictions: {len(incorrect_df)}\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Correct prediction percentage: "
        f"{len(correct_df) / len(analysis_df) * 100:.2f}%\n"
    )

    file.write(
        f"Incorrect prediction percentage: "
        f"{len(incorrect_df) / len(analysis_df) * 100:.2f}%\n\n"
    )

    file.write("-" * 70 + "\n")
    file.write("2. CLASSIFICATION REPORT\n")
    file.write("-" * 70 + "\n\n")

    file.write(class_report)
    file.write("\n")

    file.write("-" * 70 + "\n")
    file.write("3. CLASS-WISE ERROR SUMMARY\n")
    file.write("-" * 70 + "\n\n")

    file.write(
        class_summary_df.to_string(index=False)
    )

    file.write("\n\n")

    file.write("-" * 70 + "\n")
    file.write("4. COMMON SENTIMENT CONFUSION PATTERNS\n")
    file.write("-" * 70 + "\n\n")

    if confusion_pairs_df.empty:

        file.write("No incorrect class predictions found.\n")

    else:

        file.write(
            confusion_pairs_df.to_string(index=False)
        )

    file.write("\n\n")

    file.write("-" * 70 + "\n")
    file.write("5. SAMPLE MISCLASSIFIED EXAMPLES\n")
    file.write("-" * 70 + "\n\n")

    if incorrect_df.empty:

        file.write("No misclassified examples found.\n")

    else:

        # Display up to five examples for each actual sentiment.
        for label in LABELS:

            examples = incorrect_df[
                incorrect_df["actual_sentiment"] == label
            ].head(5)

            if not examples.empty:

                file.write(
                    f"\nActual sentiment: {label.upper()}\n"
                )

                for _, row in examples.iterrows():

                    file.write(
                        f"\nAspect: {row['Aspect Term']}\n"
                    )

                    file.write(
                        f"Actual: {row['actual_sentiment']}\n"
                    )

                    file.write(
                        f"Predicted: {row['predicted_sentiment']}\n"
                    )

                    file.write(
                        f"Sentence: {row['Sentence']}\n"
                    )

    file.write("\n")
    file.write("-" * 70 + "\n")
    file.write("6. INTERPRETATION AND LIMITATIONS\n")
    file.write("-" * 70 + "\n\n")

    file.write(
        "This analysis examines the errors made by the selected "
        "Baseline Linear SVM on the original held-out test set.\n\n"
    )

    file.write(
        "Class-wise error statistics help identify which sentiment "
        "categories are more difficult for the model to classify.\n\n"
    )

    file.write(
        "The conflict class has very few test examples. Its evaluation "
        "metrics should therefore be interpreted cautiously.\n\n"
    )

    file.write(
        "Neutral sentiment may be confused with positive or negative "
        "sentiment when the review language contains mixed or subtle "
        "opinions.\n\n"
    )

    file.write(
        "The project uses provided aspect terms. It classifies the "
        "sentiment associated with a given aspect; it does not "
        "independently extract aspect terms.\n\n"
    )

    file.write(
        "This is an academic machine learning experiment and is not "
        "a clinically or commercially validated system.\n"
    )


# ============================================================
# 14. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOVERALL PERFORMANCE")

print("Total test records:", len(analysis_df))
print("Correct predictions:", len(correct_df))
print("Incorrect predictions:", len(incorrect_df))
print(f"Accuracy: {accuracy:.4f}")

print("\nCLASS-WISE ERROR SUMMARY")

print(
    class_summary_df.to_string(index=False)
)

print("\nMOST COMMON CONFUSION PATTERNS")

if confusion_pairs_df.empty:
    print("No confusion patterns found.")
else:
    print(
        confusion_pairs_df.to_string(index=False)
    )

print("\nFILES SAVED")

print(report_file)
print(misclassified_file)
print(all_predictions_file)
print(confusion_plot_file)

print("\nStage 6 finished.")
print("The selected final model has not been changed.")
