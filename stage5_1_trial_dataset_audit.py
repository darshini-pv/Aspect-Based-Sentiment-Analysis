
# ============================================================
# ASPECT-BASED SENTIMENT ANALYSIS
# STAGE 5.1: TRIAL DATASET AUDIT
# ============================================================

import os
import pandas as pd
import numpy as np

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_FOLDER = (
    r"C:\Users\manoj\OneDrive\Desktop\Aspect-Based Sentiment Analysis"
)

DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

TRAIN_PATH = os.path.join(DATA_FOLDER, "Laptop_Train_v2.csv")
TRIAL_PATH = os.path.join(DATA_FOLDER, "laptops-trial.csv")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("=" * 65)
print("STAGE 5.1: TRIAL DATASET AUDIT")
print("=" * 65)

# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        "Main training dataset not found: " + TRAIN_PATH
    )

if not os.path.exists(TRIAL_PATH):
    raise FileNotFoundError(
        "Trial dataset not found: " + TRIAL_PATH
    )

train_df = pd.read_csv(TRAIN_PATH)
trial_df = pd.read_csv(TRIAL_PATH)

print("\nMain training dataset shape:", train_df.shape)
print("Trial dataset shape:", trial_df.shape)

print("\nMain training columns:")
print(train_df.columns.tolist())

print("\nTrial dataset columns:")
print(trial_df.columns.tolist())

# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id",
    "Sentence",
    "Aspect Term",
    "polarity",
    "from",
    "to"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            "Required column missing from training data: "
            + column
        )

    if column not in trial_df.columns:
        raise ValueError(
            "Required column missing from trial data: "
            + column
        )

print("\nRequired columns verified.")

# ============================================================
# 4. CHECK SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "-" * 65)
print("TRIAL DATASET SENTIMENT DISTRIBUTION")
print("-" * 65)

trial_distribution = trial_df["polarity"].value_counts()

print(trial_distribution)

print("\nTrial sentiment percentages:")

print(
    trial_df["polarity"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "-" * 65)
print("MISSING VALUE CHECK")
print("-" * 65)

print("\nTraining dataset:")
print(train_df.isnull().sum())

print("\nTrial dataset:")
print(trial_df.isnull().sum())

# ============================================================
# 6. CHECK EXACT DUPLICATE RECORDS
# ============================================================

print("\n" + "-" * 65)
print("DUPLICATE RECORD CHECK")
print("-" * 65)

print(
    "Training duplicate records:",
    train_df.duplicated().sum()
)

print(
    "Trial duplicate records:",
    trial_df.duplicated().sum()
)

# ============================================================
# 7. CHECK SENTENCE OVERLAP
# ============================================================

print("\n" + "-" * 65)
print("SENTENCE OVERLAP CHECK")
print("-" * 65)

train_sentences = set(
    train_df["Sentence"].dropna().astype(str).str.strip()
)

trial_sentences = set(
    trial_df["Sentence"].dropna().astype(str).str.strip()
)

sentence_overlap = train_sentences.intersection(
    trial_sentences
)

print(
    "Unique training sentences:",
    len(train_sentences)
)

print(
    "Unique trial sentences:",
    len(trial_sentences)
)

print(
    "Overlapping sentence texts:",
    len(sentence_overlap)
)

# ============================================================
# 8. CHECK ASPECT-LEVEL OVERLAP
# ============================================================

print("\n" + "-" * 65)
print("ASPECT-LEVEL OVERLAP CHECK")
print("-" * 65)

aspect_columns = [
    "Sentence",
    "Aspect Term",
    "polarity"
]

train_aspects = set(
    map(
        tuple,
        train_df[aspect_columns]
        .fillna("")
        .astype(str)
        .values
    )
)

trial_aspects = set(
    map(
        tuple,
        trial_df[aspect_columns]
        .fillna("")
        .astype(str)
        .values
    )
)

aspect_overlap = train_aspects.intersection(
    trial_aspects
)

print(
    "Training aspect-level records:",
    len(train_aspects)
)

print(
    "Trial aspect-level records:",
    len(trial_aspects)
)

print(
    "Overlapping aspect-level records:",
    len(aspect_overlap)
)

# ============================================================
# 9. CHECK SENTENCE ID OVERLAP
# ============================================================

print("\n" + "-" * 65)
print("SENTENCE ID OVERLAP CHECK")
print("-" * 65)

train_ids = set(
    train_df["id"].dropna().astype(str)
)

trial_ids = set(
    trial_df["id"].dropna().astype(str)
)

id_overlap = train_ids.intersection(trial_ids)

print("Training sentence IDs:", len(train_ids))
print("Trial sentence IDs:", len(trial_ids))
print("Overlapping sentence IDs:", len(id_overlap))

# ============================================================
# 10. CHECK WHETHER TRIAL DATA IS LABELLED
# ============================================================

print("\n" + "-" * 65)
print("TRIAL DATASET LABEL STATUS")
print("-" * 65)

if "polarity" in trial_df.columns:

    labelled_records = trial_df["polarity"].notna().sum()

    print(
        "Records with sentiment labels:",
        labelled_records
    )

    print(
        "Records without sentiment labels:",
        trial_df["polarity"].isna().sum()
    )

else:

    print("Trial dataset does not contain a polarity column.")

# ============================================================
# 11. FINAL AUDIT SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_FOLDER,
    "stage5_1_trial_dataset_audit.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 5.1: TRIAL DATASET AUDIT\n")
    file.write("=" * 65 + "\n\n")

    file.write(
        f"Training dataset shape: {train_df.shape}\n"
    )

    file.write(
        f"Trial dataset shape: {trial_df.shape}\n\n"
    )

    file.write("Trial sentiment distribution:\n")
    file.write(
        trial_distribution.to_string()
    )

    file.write("\n\n")

    file.write(
        f"Overlapping sentence texts: "
        f"{len(sentence_overlap)}\n"
    )

    file.write(
        f"Overlapping aspect-level records: "
        f"{len(aspect_overlap)}\n"
    )

    file.write(
        f"Overlapping sentence IDs: "
        f"{len(id_overlap)}\n"
    )

print("\nAudit summary saved:")
print(summary_path)

print("\n" + "=" * 65)
print("STAGE 5.1 COMPLETED SUCCESSFULLY")
print("=" * 65)
