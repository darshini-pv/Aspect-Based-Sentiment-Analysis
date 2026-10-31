
# ============================================================
# PROJECT 2: ASPECT-BASED SENTIMENT ANALYSIS
# STAGE 1: DATASET INSPECTION AND VERIFICATION
# ============================================================

import os
import pandas as pd

# ------------------------------------------------------------
# STEP 1: SET DATASET FOLDER PATH
# ------------------------------------------------------------

DATA_FOLDER = r"C:\Users\manoj\OneDrive\Desktop\Aspect-Based Sentiment Analysis"

print("=" * 70)
print("STAGE 1: DATASET INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# STEP 2: VERIFY THE DATASET FOLDER
# ------------------------------------------------------------

if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_FOLDER}\n"
        "Please update DATA_FOLDER with your actual folder path."
    )

print("\nDataset folder found successfully.")

# ------------------------------------------------------------
# STEP 3: LIST ALL FILES
# ------------------------------------------------------------

files = os.listdir(DATA_FOLDER)

print("\nFiles available in the dataset folder:")

for file in files:
    print(" -", file)

# ------------------------------------------------------------
# STEP 4: IDENTIFY CSV FILES
# ------------------------------------------------------------

csv_files = [
    file for file in files
    if file.lower().endswith(".csv")
]

print("\nCSV files found:", len(csv_files))

# ------------------------------------------------------------
# STEP 5: INSPECT EACH CSV FILE
# ------------------------------------------------------------

for file in csv_files:

    file_path = os.path.join(DATA_FOLDER, file)

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    try:

        df = pd.read_csv(file_path)

        # Dataset dimensions
        print("\nDataset shape:")
        print(df.shape)

        # Column names
        print("\nColumn names:")
        print(df.columns.tolist())

        # First five rows
        print("\nFirst five records:")
        print(df.head())

        # Data types
        print("\nData types:")
        print(df.dtypes)

        # Missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Duplicate records
        print("\nDuplicate records:")
        print(df.duplicated().sum())

        # Unique values for categorical/small columns
        print("\nUnique value counts for selected columns:")

        for column in df.columns:

            if df[column].nunique() <= 20:

                print(f"\nColumn: {column}")
                print(df[column].value_counts(dropna=False))

    except Exception as error:

        print("\nUnable to inspect this file.")
        print("Error:", error)

# ------------------------------------------------------------
# STEP 6: STAGE 1 COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 1 DATASET INSPECTION COMPLETED")
print("=" * 70)
