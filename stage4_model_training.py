
# ============================================================
# STAGE 4: MACHINE LEARNING MODEL TRAINING
# Project: Aspect-Based Sentiment Analysis
# Models: Logistic Regression and Linear SVM
# ============================================================

import os
import time
import joblib
import numpy as np

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = (
    r"C:\Users\manoj\OneDrive\Desktop\Aspect-Based Sentiment Analysis"
)

MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD STAGE 3 FEATURES AND LABELS
# ------------------------------------------------------------

print("=" * 65)
print("STAGE 4: MACHINE LEARNING MODEL TRAINING")
print("=" * 65)

X_train = load_npz(
    os.path.join(MODELS_FOLDER, "X_train_tfidf.npz")
)

X_test = load_npz(
    os.path.join(MODELS_FOLDER, "X_test_tfidf.npz")
)

y_train = np.load(
    os.path.join(MODELS_FOLDER, "y_train.npy"),
    allow_pickle=False
)

y_test = np.load(
    os.path.join(MODELS_FOLDER, "y_test.npy"),
    allow_pickle=False
)

print("\nStage 3 data loaded successfully.")

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))


# ------------------------------------------------------------
# 3. VERIFY DATA
# ------------------------------------------------------------

if X_train.shape[0] != len(y_train):
    raise ValueError("Training feature and label counts do not match.")

if X_test.shape[0] != len(y_test):
    raise ValueError("Testing feature and label counts do not match.")

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and testing feature dimensions differ.")

if not np.isfinite(X_train.data).all():
    raise ValueError("Invalid values found in training features.")

if not np.isfinite(X_test.data).all():
    raise ValueError("Invalid values found in testing features.")

expected_classes = {
    "positive", "negative", "neutral", "conflict"
}

if set(np.unique(y_train)) != expected_classes:
    raise ValueError("Unexpected sentiment classes in training data.")

print("\nData verification successful.")

print("\nTraining sentiment distribution:")
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"{label}: {count}")


# ------------------------------------------------------------
# 4. INITIALIZE LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)


# ------------------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("TRAINING MODEL 1: LOGISTIC REGRESSION")
print("-" * 65)

start_time = time.time()

logistic_model.fit(X_train, y_train)

logistic_time = time.time() - start_time

print("Logistic Regression training completed.")
print(f"Training time: {logistic_time:.4f} seconds")
print("Classes:", logistic_model.classes_)
print("Iterations:", logistic_model.n_iter_)


# ------------------------------------------------------------
# 6. INITIALIZE LINEAR SVM
# ------------------------------------------------------------

svm_model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)


# ------------------------------------------------------------
# 7. TRAIN LINEAR SVM
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("TRAINING MODEL 2: LINEAR SVM")
print("-" * 65)

start_time = time.time()

svm_model.fit(X_train, y_train)

svm_time = time.time() - start_time

print("Linear SVM training completed.")
print(f"Training time: {svm_time:.4f} seconds")
print("Classes:", svm_model.classes_)
print("Iterations:", svm_model.n_iter_)


# ------------------------------------------------------------
# 8. SAVE TRAINED MODELS
# ------------------------------------------------------------

logistic_path = os.path.join(
    MODELS_FOLDER,
    "logistic_regression_model.joblib"
)

svm_path = os.path.join(
    MODELS_FOLDER,
    "linear_svm_model.joblib"
)

joblib.dump(logistic_model, logistic_path)
joblib.dump(svm_model, svm_path)

print("\nBoth trained models saved successfully.")


# ------------------------------------------------------------
# 9. SAVE TRAINING SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    RESULTS_FOLDER,
    "stage4_model_training_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4: MODEL TRAINING SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Training records: {X_train.shape[0]}\n")
    file.write(f"Testing records: {X_test.shape[0]}\n")
    file.write(f"TF-IDF features: {X_train.shape[1]}\n\n")

    file.write("Model 1: Logistic Regression\n")
    file.write("C = 1.0\n")
    file.write("class_weight = balanced\n")
    file.write("max_iter = 2000\n")
    file.write(f"Training time = {logistic_time:.4f} seconds\n")
    file.write(f"Iterations = {logistic_model.n_iter_}\n\n")

    file.write("Model 2: Linear SVM\n")
    file.write("C = 1.0\n")
    file.write("class_weight = balanced\n")
    file.write("max_iter = 2000\n")
    file.write(f"Training time = {svm_time:.4f} seconds\n")
    file.write(f"Iterations = {svm_model.n_iter_}\n\n")

    file.write("Sentiment classes:\n")
    file.write(", ".join(logistic_model.classes_))


# ------------------------------------------------------------
# 10. COMPLETION
# ------------------------------------------------------------

print("\nSaved files:")
print("1. models/logistic_regression_model.joblib")
print("2. models/linear_svm_model.joblib")
print("3. results/stage4_model_training_summary.txt")

print("\n" + "=" * 65)
print("STAGE 4 COMPLETED SUCCESSFULLY")
print("=" * 65)
