
# ============================================================
# STAGE 5.2: FEATURE ENGINEERING AND HYPERPARAMETER TUNING
# Project: Aspect-Based Sentiment Analysis
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from scipy.sparse import load_npz


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

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 70)
print("STAGE 5.2: FEATURE ENGINEERING AND HYPERPARAMETER TUNING")
print("=" * 70)


# ------------------------------------------------------------
# 3. LOAD ORIGINAL DATASET
# ------------------------------------------------------------

print("\nLoading original labelled dataset...")

df = pd.read_csv(DATA_FILE)

required_columns = [
    "id", "Sentence", "Aspect Term", "polarity"
]

if not all(column in df.columns for column in required_columns):
    raise ValueError("Required dataset columns are missing.")

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 4. PREPARE ASPECT-AWARE TEXT
# ------------------------------------------------------------

print("\nPreparing aspect-aware text...")

df["combined_text"] = (
    df["Sentence"].astype(str)
    + " targetaspect "
    + df["Aspect Term"].astype(str)
)

X = df["combined_text"]
y = df["polarity"].astype(str)

# Retain the original ID type and row order.
groups = df["id"]

print("Aspect-aware text prepared.")
print("\nExample:")
print(X.iloc[0])


# ------------------------------------------------------------
# 5. RECREATE ORIGINAL STAGE 3 SPLIT
# ------------------------------------------------------------

print("\nReconstructing original Stage 3 split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train_text = X.iloc[train_indices].reset_index(drop=True)
X_test_text = X.iloc[test_indices].reset_index(drop=True)

y_train_text = y.iloc[train_indices].reset_index(drop=True)
y_test_text = y.iloc[test_indices].reset_index(drop=True)

train_groups = groups.iloc[train_indices].reset_index(drop=True)
test_groups = groups.iloc[test_indices].reset_index(drop=True)

overlap = set(train_groups).intersection(set(test_groups))

print("Training records:", len(X_train_text))
print("Testing records:", len(X_test_text))
print("Training sentence IDs:", train_groups.nunique())
print("Testing sentence IDs:", test_groups.nunique())
print("Overlapping sentence IDs:", len(overlap))

if len(overlap) != 0:
    raise ValueError("Sentence ID leakage detected.")

if len(X_train_text) != 1880 or len(X_test_text) != 478:
    raise ValueError(
        "The reconstructed split does not match Stage 3."
    )


# ------------------------------------------------------------
# 6. LOAD SAVED STAGE 3 FILES
# ------------------------------------------------------------

print("\nLoading saved Stage 3 files...")

X_train_saved = load_npz(
    os.path.join(MODELS_FOLDER, "X_train_tfidf.npz")
)

X_test_saved = load_npz(
    os.path.join(MODELS_FOLDER, "X_test_tfidf.npz")
)

y_train_saved = np.load(
    os.path.join(MODELS_FOLDER, "y_train.npy"),
    allow_pickle=False
)

y_test_saved = np.load(
    os.path.join(MODELS_FOLDER, "y_test.npy"),
    allow_pickle=False
)

# Verify row counts and labels.
if X_train_saved.shape[0] != len(X_train_text):
    raise ValueError("Saved training matrix row count mismatch.")

if X_test_saved.shape[0] != len(X_test_text):
    raise ValueError("Saved testing matrix row count mismatch.")

if not np.array_equal(
    y_train_saved.astype(str),
    y_train_text.to_numpy(dtype=str)
):
    raise ValueError("Training labels do not match Stage 3.")

if not np.array_equal(
    y_test_saved.astype(str),
    y_test_text.to_numpy(dtype=str)
):
    raise ValueError("Testing labels do not match Stage 3.")

print("Saved Stage 3 split dimensions and labels verified.")
print("Training matrix:", X_train_saved.shape)
print("Testing matrix:", X_test_saved.shape)

# Original saved matrices are retained unchanged.
# Cross-validation below uses raw text and fits TF-IDF
# separately inside each fold.


# ------------------------------------------------------------
# 7. PREPARE GROUPED CROSS-VALIDATION
# ------------------------------------------------------------

print("\nPreparing grouped cross-validation...")

cv = GroupKFold(n_splits=3)

groups_array = train_groups.to_numpy()

labels = [
    "conflict",
    "negative",
    "neutral",
    "positive"
]


# ------------------------------------------------------------
# 8. DEFINE HYPERPARAMETER CONFIGURATIONS
# ------------------------------------------------------------

parameter_combinations = []

for ngram_range in [(1, 1), (1, 2)]:

    for C in [0.1, 1, 10]:

        for class_weight in ["balanced", None]:

            parameter_combinations.append({
                "ngram_range": ngram_range,
                "C": C,
                "class_weight": class_weight
            })

print("Total configurations:", len(parameter_combinations))


# ------------------------------------------------------------
# 9. RUN GROUPED CROSS-VALIDATION
# ------------------------------------------------------------

results = []

print("\nStarting hyperparameter tuning...")
print("Scoring metric: Macro-F1")
print("Cross-validation folds: 3")
print("Original test set: NOT used for tuning")

for config_number, params in enumerate(
    parameter_combinations,
    start=1
):

    fold_scores = []

    print("\n" + "-" * 60)
    print(
        "Configuration",
        config_number,
        "of",
        len(parameter_combinations)
    )
    print("Parameters:", params)

    for fold_number, (train_idx, validation_idx) in enumerate(
        cv.split(
            X_train_text,
            y_train_text,
            groups=groups_array
        ),
        start=1
    ):

        X_fold_train = X_train_text.iloc[train_idx]
        X_fold_validation = X_train_text.iloc[validation_idx]

        y_fold_train = y_train_text.iloc[train_idx]
        y_fold_validation = y_train_text.iloc[validation_idx]

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=None,
                    ngram_range=params["ngram_range"],
                    max_features=10000,
                    min_df=2
                )
            ),
            (
                "svm",
                LinearSVC(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    max_iter=5000,
                    random_state=42
                )
            )
        ])

        pipeline.fit(
            X_fold_train,
            y_fold_train
        )

        predictions = pipeline.predict(
            X_fold_validation
        )

        score = f1_score(
            y_fold_validation,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0
        )

        fold_scores.append(score)

        print(
            "Fold",
            fold_number,
            "Macro-F1:",
            round(score, 4)
        )

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    results.append({
        "N-gram Range": str(params["ngram_range"]),
        "C": params["C"],
        "Class Weight": str(params["class_weight"]),
        "Mean Macro-F1": mean_score,
        "Standard Deviation": std_score
    })

    print("Mean Macro-F1:", round(mean_score, 4))
    print("Standard Deviation:", round(std_score, 4))


# ------------------------------------------------------------
# 10. SELECT BEST CONFIGURATION
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean Macro-F1",
    ascending=False
).reset_index(drop=True)

best_result = results_df.iloc[0]

if best_result["N-gram Range"] == "(1, 1)":
    best_ngram = (1, 1)
else:
    best_ngram = (1, 2)

best_C = float(best_result["C"])

if best_result["Class Weight"] == "None":
    best_class_weight = None
else:
    best_class_weight = "balanced"


print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nBEST CONFIGURATION")
print("N-gram range:", best_ngram)
print("C:", best_C)
print("Class weight:", best_class_weight)
print(
    "Mean CV Macro-F1:",
    round(best_result["Mean Macro-F1"], 4)
)


# ------------------------------------------------------------
# 11. TRAIN FINAL TUNED MODEL
# ------------------------------------------------------------

print("\nTraining final tuned Linear SVM...")

best_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=best_ngram,
            max_features=10000,
            min_df=2
        )
    ),
    (
        "svm",
        LinearSVC(
            C=best_C,
            class_weight=best_class_weight,
            max_iter=5000,
            random_state=42
        )
    )
])

best_pipeline.fit(
    X_train_text,
    y_train_text
)

print("Final tuned model trained successfully.")


# ------------------------------------------------------------
# 12. SAVE MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODELS_FOLDER,
    "tuned_linear_svm_model.joblib"
)

joblib.dump(
    best_pipeline,
    model_path
)


# ------------------------------------------------------------
# 13. SAVE RESULTS
# ------------------------------------------------------------

results_path = os.path.join(
    RESULTS_FOLDER,
    "stage5_2_tuning_results.txt"
)

with open(results_path, "w", encoding="utf-8") as file:

    file.write(
        "STAGE 5.2: FEATURE ENGINEERING AND HYPERPARAMETER TUNING\n"
    )
    file.write("=" * 65 + "\n\n")

    file.write("Original Stage 3 split verified.\n")
    file.write("Training records: 1880\n")
    file.write("Testing records: 478\n")
    file.write("Overlapping sentence IDs: 0\n\n")

    file.write("Grouped cross-validation: 3 folds\n")
    file.write("Scoring metric: Macro-F1\n")
    file.write("Stop words: None\n\n")

    file.write("ALL CONFIGURATION RESULTS:\n\n")
    file.write(results_df.to_string(index=False))

    file.write("\n\nBEST CONFIGURATION:\n")
    file.write(f"N-gram range: {best_ngram}\n")
    file.write(f"C: {best_C}\n")
    file.write(f"Class weight: {best_class_weight}\n")

    file.write(
        f"Mean CV Macro-F1: {best_result['Mean Macro-F1']:.4f}\n"
    )

    file.write(
        f"Standard deviation: {best_result['Standard Deviation']:.4f}\n"
    )

    file.write("\nFinal tuned model saved successfully.\n")


# ------------------------------------------------------------
# 14. COMPLETION
# ------------------------------------------------------------

print("\nSaved files:")
print("1. models/tuned_linear_svm_model.joblib")
print("2. results/stage5_2_tuning_results.txt")

print("\n" + "=" * 70)
print("STAGE 5.2 COMPLETED SUCCESSFULLY")
print("=" * 70)
