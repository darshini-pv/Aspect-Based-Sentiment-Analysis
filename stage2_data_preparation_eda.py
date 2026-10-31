

# ============================================================
# PROJECT 2: ASPECT-BASED SENTIMENT ANALYSIS
# STAGE 2: DATA PREPARATION AND EXPLORATORY DATA ANALYSIS
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1: DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Aspect-Based Sentiment Analysis"
)

DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_FOLDER, "Laptop_Train_v2.csv")
TRIAL_FILE = os.path.join(DATA_FOLDER, "laptops-trial.csv")
TEST_A_FILE = os.path.join(DATA_FOLDER, "Laptops_Test_Data_PhaseA.csv")
TEST_B_FILE = os.path.join(DATA_FOLDER, "Laptops_Test_Data_PhaseB.csv")

print("=" * 70)
print("STAGE 2: DATA PREPARATION AND EDA")
print("=" * 70)

# ------------------------------------------------------------
# STEP 2: LOAD THE DATASETS
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
trial_df = pd.read_csv(TRIAL_FILE)
test_a = pd.read_csv(TEST_A_FILE)
test_b = pd.read_csv(TEST_B_FILE)

print("\nAll four CSV files loaded successfully.")

print("\nTraining dataset shape:", train_df.shape)
print("Trial dataset shape:", trial_df.shape)
print("Phase A test shape:", test_a.shape)
print("Phase B test shape:", test_b.shape)

# ------------------------------------------------------------
# STEP 3: CHECK SENTIMENT DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SENTIMENT DISTRIBUTION")
print("=" * 70)

sentiment_counts = train_df["polarity"].value_counts()

print(sentiment_counts)

sentiment_percentages = (
    train_df["polarity"].value_counts(normalize=True) * 100
)

print("\nSentiment percentages:")
print(sentiment_percentages.round(2))

sentiment_summary = pd.DataFrame({
    "Count": sentiment_counts,
    "Percentage": sentiment_percentages
})

sentiment_summary.to_csv(
    os.path.join(RESULTS_FOLDER, "sentiment_distribution.csv")
)

# ------------------------------------------------------------
# STEP 4: VISUALIZE SENTIMENT DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sentiment_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Sentiment Distribution in Laptop Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_FOLDER, "sentiment_distribution.png")
)

plt.show()

# ------------------------------------------------------------
# STEP 5: CHECK DATA QUALITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nDuplicate complete records:")
print(train_df.duplicated().sum())

print("\nUnique sentence IDs:", train_df["id"].nunique())

print("Unique sentence texts:", train_df["Sentence"].nunique())

print("Unique aspect terms:", train_df["Aspect Term"].nunique())

# ------------------------------------------------------------
# STEP 6: CHECK ASPECT ANNOTATION POSITIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ASPECT POSITION VALIDATION")
print("=" * 70)

# The 'from' position is the starting character.
# The 'to' position is the ending character (exclusive).

span_matches = []

for _, row in train_df.iterrows():

    sentence = str(row["Sentence"])
    aspect = str(row["Aspect Term"])

    start = int(row["from"])
    end = int(row["to"])

    extracted_aspect = sentence[start:end]

    span_matches.append(extracted_aspect == aspect)

train_df["span_valid"] = span_matches

valid_spans = train_df["span_valid"].sum()
invalid_spans = (~train_df["span_valid"]).sum()

print("Valid aspect annotations:", valid_spans)
print("Invalid aspect annotations:", invalid_spans)

# Save invalid examples for inspection if any exist.

invalid_examples = train_df[
    train_df["span_valid"] == False
]

invalid_examples.to_csv(
    os.path.join(RESULTS_FOLDER, "invalid_aspect_spans.csv"),
    index=False
)

# ------------------------------------------------------------
# STEP 7: FIND MOST FREQUENT ASPECT TERMS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MOST FREQUENT ASPECT TERMS")
print("=" * 70)

aspect_counts = train_df["Aspect Term"].value_counts()

print(aspect_counts.head(15))

aspect_counts.head(20).to_csv(
    os.path.join(RESULTS_FOLDER, "top_aspect_terms.csv")
)

# ------------------------------------------------------------
# STEP 8: CHECK REPEATED SENTENCES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REPEATED SENTENCE ANALYSIS")
print("=" * 70)

sentence_counts = train_df["Sentence"].value_counts()

repeated_sentences = sentence_counts[
    sentence_counts > 1
]

print("Total unique sentences:", train_df["Sentence"].nunique())

print("Sentences with multiple aspect annotations:",
      len(repeated_sentences))

print("\nExamples:")
print(repeated_sentences.head(10))

# ------------------------------------------------------------
# STEP 9: COMPARE PHASE A AND PHASE B
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST DATASET COMPARISON")
print("=" * 70)

same_test_data = (
    test_a.reset_index(drop=True)
    .equals(test_b.reset_index(drop=True))
)

print("Are Phase A and Phase B identical?", same_test_data)

print("\nPhase A columns:", test_a.columns.tolist())
print("Phase B columns:", test_b.columns.tolist())

# ------------------------------------------------------------
# STEP 10: SAVE DATA QUALITY SUMMARY
# ------------------------------------------------------------

summary = [
    "STAGE 2 DATA PREPARATION AND EDA SUMMARY",
    "=" * 50,
    f"Training records: {len(train_df)}",
    f"Unique sentence IDs: {train_df['id'].nunique()}",
    f"Unique sentence texts: {train_df['Sentence'].nunique()}",
    f"Unique aspect terms: {train_df['Aspect Term'].nunique()}",
    f"Duplicate records: {train_df.duplicated(subset=['id', 'Sentence', 'Aspect Term', 'polarity', 'from', 'to']).sum()}",
    f"Missing values: {train_df.isnull().sum().sum()}",
    f"Valid aspect spans: {valid_spans}",
    f"Invalid aspect spans: {invalid_spans}",
    f"Repeated sentence texts: {len(repeated_sentences)}",
    f"Phase A and B identical: {same_test_data}",
    "",
    "SENTIMENT COUNTS:",
    sentiment_counts.to_string(),
    "",
    "STAGE 2 INSPECTION COMPLETED"
]

summary_path = os.path.join(
    RESULTS_FOLDER,
    "stage2_eda_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("\n".join(summary))

print("\nSummary saved to:", summary_path)

print("\n" + "=" * 70)
print("STAGE 2 COMPLETED")
print("=" * 70)
