# =============================================================================
# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
# =============================================================================
# Phase 2: Data Cleaning & Preprocessing
# Author   : Senior Data Scientist & Python Mentor
# Date     : November 2026
# Module   : notebooks/phase2_data_cleaning.py
# =============================================================================

# -----------------------------------------------------------------------------
# WHY DO WE IMPORT THESE LIBRARIES?
# - os: For cross-platform absolute path resolution.
# - pandas: For DataFrame operations, imputation, type casting, and validation.
# - numpy: For numerical representations.
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

# =============================================================================
# STEP 1: LOAD RAW DATASET WITH PROPER SCHEMA (From Phase 1)
# =============================================================================
# WHY: We load the raw file with the column names established in Phase 1,
# ensuring '?' is converted to NaN at the time of reading.
# =============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dataset_path = os.path.join(project_root, "dataset", "processed.cleveland.data")
cleaned_dataset_path = os.path.join(project_root, "dataset", "cleaned_cardiopredict.csv")

column_names = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope", "CA", "Thal", "Target"
]

print("=" * 70)
print("   CardioPredict — Phase 2: Complete Data Cleaning & Preprocessing")
print("=" * 70)

print("\n📂 STEP 1: Loading Raw Cleveland Dataset")
print("-" * 70)

heart_df = pd.read_csv(
    raw_dataset_path,
    header=None,
    names=column_names,
    na_values="?"
)

# Binarize Target: 0 = Healthy, 1 = Heart Disease
heart_df["Target"] = heart_df["Target"].apply(lambda x: 1 if x > 0 else 0)

print(f"✅ Loaded raw dataset successfully: {heart_df.shape[0]} rows, {heart_df.shape[1]} columns.")

# =============================================================================
# STEP 2: MISSING VALUE AUDIT & IDENTIFICATION
# =============================================================================
# WHY: In clinical data, missing values are rarely accidental; they occur due
# to clinical contraindications, patient refusal, or procedural constraints.
# Before deciding how to handle them, we must quantify where and why they exist.
# =============================================================================
print("\n🔍 STEP 2: Missing Value Identification & Audit")
print("-" * 70)

missing_count = heart_df.isnull().sum()
missing_percentage = (missing_count / len(heart_df)) * 100

missing_report = pd.DataFrame({
    "Feature": heart_df.columns,
    "Missing Count": missing_count.values,
    "Missing Percentage (%)": missing_percentage.round(2).values
})

# Display only features with missing values
missing_features = missing_report[missing_report["Missing Count"] > 0]
print(missing_features.to_string(index=False))

total_missing_cells = heart_df.isnull().sum().sum()
total_cells = heart_df.size
print(f"\nTotal Missing Cells: {total_missing_cells} out of {total_cells} ({total_missing_cells/total_cells*100:.2f}%)")

# Inspect the exact patient records containing missing values
missing_rows_df = heart_df[heart_df.isnull().any(axis=1)]
print(f"\nPatients with Missing Observations ({len(missing_rows_df)} rows):")
print(missing_rows_df[["Age", "Sex", "ChestPainType", "CA", "Thal", "Target"]].to_string())

# =============================================================================
# STEP 3: HANDLING MISSING VALUES VIA MODE IMPUTATION
# =============================================================================
# WHY WE USE MODE IMPUTATION:
# 1. 'CA' (Number of major vessels) is a discrete categorical count (0, 1, 2, 3).
#    Mean imputation would yield non-integer values (e.g. 0.67 vessels), which
#    is clinically invalid.
# 2. 'Thal' (Thallium test) is a nominal categorical variable (3, 6, 7).
#    Mean imputation is mathematically meaningless for nominal categories.
# 3. Mode (most frequent value) is the statistically sound, standard method for
#    categorical and discrete medical attributes.
# 4. Deleting rows would sacrifice 6 valuable patient cases (~2% of data) in a
#    modest-sized sample of 303 patients.
# =============================================================================
print("\n🛠️  STEP 3: Handling Missing Values (Mode Imputation)")
print("-" * 70)

ca_mode = heart_df["CA"].mode()[0]
thal_mode = heart_df["Thal"].mode()[0]

print(f"   Calculated Mode for 'CA'   : {ca_mode} (Represents 0 colored vessels - most common)")
print(f"   Calculated Mode for 'Thal' : {thal_mode} (Represents 3.0: Normal blood flow - most common)")

# Perform imputation
heart_df["CA"] = heart_df["CA"].fillna(ca_mode)
heart_df["Thal"] = heart_df["Thal"].fillna(thal_mode)

remaining_missing = heart_df.isnull().sum().sum()
print(f"\n✅ Missing values after mode imputation: {remaining_missing}")

# =============================================================================
# STEP 4: DUPLICATE ROW CHECK & REMOVAL
# =============================================================================
# WHY: Duplicate rows skew machine learning models, causing artificial data
# leakage between training and testing folds, inflating performance metrics.
# =============================================================================
print("\n👥 STEP 4: Duplicate Records Check")
print("-" * 70)

duplicate_count = heart_df.duplicated().sum()
print(f"   Duplicate rows detected: {duplicate_count}")

if duplicate_count > 0:
    heart_df = heart_df.drop_duplicates()
    print(f"   Removed {duplicate_count} duplicate row(s).")
else:
    print("   ✅ No duplicate patient records found. Data integrity intact.")

# =============================================================================
# STEP 5: DOMAIN INTEGRITY & INCONSISTENCY VALIDATION
# =============================================================================
# WHY: We must verify that clinical features fall within physiologically valid
# ranges and adhere to the UCI encoding guidelines.
# =============================================================================
print("\n🩺 STEP 5: Clinical Domain Range & Inconsistency Validation")
print("-" * 70)

validation_rules = {
    "Age": (heart_df["Age"].between(1, 120).all(), "Between 1 and 120 years"),
    "Sex": (heart_df["Sex"].isin([0, 1]).all(), "Binary {0: Female, 1: Male}"),
    "ChestPainType": (heart_df["ChestPainType"].isin([1, 2, 3, 4]).all(), "Categorical {1, 2, 3, 4}"),
    "RestingBP": (heart_df["RestingBP"].between(50, 300).all(), "Between 50 and 300 mm Hg"),
    "Cholesterol": (heart_df["Cholesterol"].between(50, 700).all(), "Between 50 and 700 mg/dl"),
    "FastingBS": (heart_df["FastingBS"].isin([0, 1]).all(), "Binary {0: <=120, 1: >120 mg/dl}"),
    "RestingECG": (heart_df["RestingECG"].isin([0, 1, 2]).all(), "Categorical {0, 1, 2}"),
    "MaxHR": (heart_df["MaxHR"].between(40, 250).all(), "Between 40 and 250 bpm"),
    "ExerciseAngina": (heart_df["ExerciseAngina"].isin([0, 1]).all(), "Binary {0: No, 1: Yes}"),
    "Oldpeak": ((heart_df["Oldpeak"] >= 0).all(), "Non-negative ST depression (>= 0 mm)"),
    "ST_Slope": (heart_df["ST_Slope"].isin([1, 2, 3]).all(), "Categorical {1: Up, 2: Flat, 3: Down}"),
    "CA": (heart_df["CA"].isin([0, 1, 2, 3]).all(), "Discrete count {0, 1, 2, 3}"),
    "Thal": (heart_df["Thal"].isin([3, 6, 7]).all(), "Categorical {3: Normal, 6: Fixed, 7: Reversible}"),
    "Target": (heart_df["Target"].isin([0, 1]).all(), "Binary {0: Healthy, 1: Heart Disease}")
}

all_valid = True
for feature, (is_valid, rule) in validation_rules.items():
    status = "✅ PASS" if is_valid else "❌ FAIL"
    if not is_valid:
        all_valid = False
    print(f"   {feature:<15} : {status} (Expected: {rule})")

if all_valid:
    print("\n   ✅ All clinical features strictly conform to medical domain rules!")

# =============================================================================
# STEP 6: DATA TYPE CONVERSIONS
# =============================================================================
# WHY: Due to missing values ('?'), Pandas previously parsed discrete counts
# and integer categories as float64. Now that missing values are resolved,
# we convert integer categories and counts to int64 for computational efficiency
# and semantic correctness. Oldpeak remains float64 (continuous measurement).
# =============================================================================
print("\n🔄 STEP 6: Data Type Optimization & Type Casting")
print("-" * 70)

integer_columns = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "ST_Slope", "CA", "Thal", "Target"
]

print("   Data types before casting:")
print(f"   float64 columns: {len(heart_df.select_dtypes(include=['float64']).columns)}")
print(f"   int64 columns  : {len(heart_df.select_dtypes(include=['int64']).columns)}")

# Cast to integer types
for col in integer_columns:
    heart_df[col] = heart_df[col].astype("int64")

print("\n   Data types after casting:")
print(heart_df.dtypes.to_string())

# =============================================================================
# STEP 7: VERIFICATION OF THE CLEANED DATASET
# =============================================================================
# WHY: Final validation check before serializing the cleaned dataset.
# =============================================================================
print("\n" + "=" * 70)
print("📊 STEP 7: Final Verification of Cleaned Dataset")
print("=" * 70)

print(f"Dataset Shape        : {heart_df.shape} (303 patients, 14 attributes)")
print(f"Missing Values Count : {heart_df.isnull().sum().sum()}")
print(f"Duplicate Rows Count : {heart_df.duplicated().sum()}")
print("\nCleaned Dataset Head (First 5 records):")
print(heart_df.head(5).to_string())

# Save cleaned dataset for all downstream phases
heart_df.to_csv(cleaned_dataset_path, index=False)
print(f"\n💾 Cleaned dataset successfully exported to:")
print(f"   {cleaned_dataset_path}")

print("\n" + "=" * 70)
print("✅ Phase 2 Complete — Data Cleaning Successfully Finished!")
print("=" * 70)
