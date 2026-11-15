# =============================================================================
# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
# =============================================================================
# Phase 1: Project Setup & Data Loading
# Author   : [Your Name]
# Date     : November 2026
# Dataset  : UCI Heart Disease - Cleveland Clinic Foundation
# Source   : https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# =============================================================================

# -----------------------------------------------------------------------------
# WHY DO WE IMPORT LIBRARIES FIRST?
# In Python, libraries are pre-written collections of functions. By importing
# them at the top, we follow PEP-8 conventions and make our dependencies clear
# to anyone reading the code for the first time.
# -----------------------------------------------------------------------------

import os           # For file path operations (cross-platform compatibility)
import pandas as pd # Pandas: The core library for tabular data manipulation
import numpy as np  # NumPy: Numerical computing, used for NaN handling

# =============================================================================
# STEP 1: VERIFY DATASET LOCATION
# =============================================================================
# WHY: Before loading data, we confirm the file exists. In a real project,
# missing data files are one of the most common early-stage bugs. This check
# prevents cryptic FileNotFoundError messages later.
# =============================================================================

# Build the path relative to the project root so the script works on any machine
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_root, "dataset", "processed.cleveland.data")

print("=" * 65)
print("   CardioPredict — Phase 1: Project Setup & Data Loading")
print("=" * 65)

print("\n📁 STEP 1: Verifying Dataset Location")
print("-" * 65)
print(f"Project Root  : {project_root}")
print(f"Dataset Path  : {dataset_path}")

if os.path.exists(dataset_path):
    file_size_kb = os.path.getsize(dataset_path) / 1024
    print(f"✅ Dataset found! File size: {file_size_kb:.2f} KB")
else:
    print("❌ ERROR: Dataset not found. Check the path and try again.")
    exit()

# =============================================================================
# STEP 2: DEFINE COLUMN NAMES
# =============================================================================
# WHY: The raw .data file has NO header row. If we load it as-is, Pandas
# would name columns 0, 1, 2 ... 13 — meaningless numbers. We define names
# based on the official UCI documentation (heart-disease.names).
# =============================================================================

print("\n📋 STEP 2: Defining Column Names")
print("-" * 65)

column_names = [
    "Age",             # Patient age in years
    "Sex",             # 1 = Male, 0 = Female
    "ChestPainType",   # 1=Typical Angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic
    "RestingBP",       # Resting blood pressure (mm Hg)
    "Cholesterol",     # Serum cholesterol (mg/dl)
    "FastingBS",       # Fasting blood sugar > 120 mg/dl (1=True, 0=False)
    "RestingECG",      # Resting electrocardiographic results (0, 1, 2)
    "MaxHR",           # Maximum heart rate achieved
    "ExerciseAngina",  # Exercise-induced angina (1=Yes, 0=No)
    "Oldpeak",         # ST depression induced by exercise relative to rest
    "ST_Slope",        # Slope of peak exercise ST segment (1, 2, 3)
    "CA",              # Number of major vessels colored by fluoroscopy (0-3)
    "Thal",            # Thalassemia: 3=Normal, 6=Fixed defect, 7=Reversible defect
    "Target"           # Diagnosis: 0=No disease, 1/2/3/4=Disease
]

print(f"✅ Defined {len(column_names)} column names:")
for i, col in enumerate(column_names, 1):
    print(f"   {i:2d}. {col}")

# =============================================================================
# STEP 3: LOAD THE DATASET
# =============================================================================
# WHY: pd.read_csv() is Pandas' powerful file loader. We pass:
#   - header=None  : Tells Pandas the file has NO header row
#   - names=...    : Assigns our custom column names
#   - na_values='?': CRITICAL — the dataset uses '?' for missing values.
#                    By telling Pandas upfront, it converts them to NaN
#                    automatically during loading (not a manual step later).
# =============================================================================

print("\n📂 STEP 3: Loading the Dataset")
print("-" * 65)

heart_df = pd.read_csv(
    dataset_path,
    header=None,          # No header row in the file
    names=column_names,   # Assign our custom column names
    na_values="?"         # Treat '?' as missing values (NaN)
)

print(f"✅ Dataset loaded successfully!")
print(f"   Rows    : {heart_df.shape[0]}")
print(f"   Columns : {heart_df.shape[1]}")

# =============================================================================
# STEP 4: BINARIZE THE TARGET VARIABLE
# =============================================================================
# WHY: The original Target has values 0, 1, 2, 3, 4.
#   - 0 = No Heart Disease (Healthy)
#   - 1, 2, 3, 4 = Varying severity of Heart Disease
#
# For binary classification (Healthy vs. Sick), we collapse all disease
# severity levels (1-4) into a single class: 1 = Heart Disease.
# This is the standard preprocessing used in all major research papers
# that use this dataset.
# =============================================================================

print("\n🎯 STEP 4: Binarizing the Target Variable")
print("-" * 65)

print("   Before binarization — unique Target values:")
print(f"   {sorted(heart_df['Target'].unique())}")

# Apply binarization: 0 stays 0; anything > 0 becomes 1
heart_df["Target"] = heart_df["Target"].apply(lambda x: 1 if x > 0 else 0)

print("   After binarization — unique Target values:")
print(f"   {sorted(heart_df['Target'].unique())}")

target_counts = heart_df["Target"].value_counts()
print(f"\n   Class Distribution:")
print(f"   0 = Healthy (No Heart Disease) : {target_counts[0]} patients")
print(f"   1 = Heart Disease              : {target_counts[1]} patients")

# =============================================================================
# STEP 5: DISPLAY FIRST 10 ROWS
# =============================================================================
# WHY: Viewing the top rows gives us a "sanity check" — we verify that
# columns are aligned correctly and values look reasonable.
# =============================================================================

print("\n" + "=" * 65)
print("📊 STEP 5: First 10 Rows of the Dataset")
print("=" * 65)
pd.set_option("display.max_columns", None)   # Show ALL columns
pd.set_option("display.width", 200)          # Wide display for readability
pd.set_option("display.float_format", "{:.1f}".format)
print(heart_df.head(10).to_string())

# =============================================================================
# STEP 6: DISPLAY LAST 10 ROWS
# =============================================================================
# WHY: The last rows may reveal data entry patterns, trailing issues,
# or anomalies at the end of the file that head() wouldn't show.
# =============================================================================

print("\n" + "=" * 65)
print("📊 STEP 6: Last 10 Rows of the Dataset")
print("=" * 65)
print(heart_df.tail(10).to_string())

# =============================================================================
# STEP 7: SHAPE OF THE DATASET
# =============================================================================
# WHY: Shape tells us the exact number of samples (rows) and features
# (columns). This is the first number every reviewer will ask about.
# =============================================================================

print("\n" + "=" * 65)
print("📐 STEP 7: Shape of the Dataset")
print("=" * 65)
print(f"   Shape  : {heart_df.shape}")
print(f"   Rows   : {heart_df.shape[0]}  (each row = 1 patient record)")
print(f"   Columns: {heart_df.shape[1]}  (13 features + 1 target)")

# =============================================================================
# STEP 8: COLUMN NAMES
# =============================================================================

print("\n" + "=" * 65)
print("🏷️  STEP 8: Column Names")
print("=" * 65)
for i, col in enumerate(heart_df.columns, 1):
    print(f"   {i:2d}. {col}")

# =============================================================================
# STEP 9: DATA TYPES
# =============================================================================
# WHY: Data types determine what operations we can perform on each column.
# - float64: decimals (most raw columns here)
# - int64:   whole numbers
# - object:  strings (should NOT appear for numeric medical data)
# Any 'object' type for a numeric column is a red flag — it means mixed
# data types or hidden characters (like '?') are still present.
# =============================================================================

print("\n" + "=" * 65)
print("🔢 STEP 9: Data Types of Each Column")
print("=" * 65)
print(heart_df.dtypes.to_string())

# =============================================================================
# STEP 10: DATASET INFORMATION (df.info())
# =============================================================================
# WHY: info() gives us the most important snapshot:
#   - Column names and their data types
#   - NON-NULL counts — immediately reveals where missing values are!
#   - Memory usage — useful for large datasets
# =============================================================================

print("\n" + "=" * 65)
print("ℹ️  STEP 10: Dataset Information (df.info())")
print("=" * 65)
heart_df.info()

# =============================================================================
# STEP 11: STATISTICAL SUMMARY (df.describe())
# =============================================================================
# WHY: describe() computes 8 summary statistics for every numeric column:
#   count, mean, std, min, 25%, 50%, 75%, max
# This one command reveals outliers (min/max), central tendency (mean/median),
# and spread (std) at a glance — invaluable for a first-pass understanding.
# =============================================================================

print("\n" + "=" * 65)
print("📈 STEP 11: Statistical Summary (df.describe())")
print("=" * 65)
print(heart_df.describe().round(2).to_string())

# =============================================================================
# STEP 12: MISSING VALUE QUICK CHECK
# =============================================================================
# WHY: This gives a preliminary count of NaN values per column —
# essential to know BEFORE cleaning (Phase 2). We want to see WHERE
# missing values are so we can plan our cleaning strategy.
# =============================================================================

print("\n" + "=" * 65)
print("❓ STEP 12: Missing Values Quick Check")
print("=" * 65)
missing_counts  = heart_df.isnull().sum()
missing_percent = (heart_df.isnull().sum() / len(heart_df)) * 100
missing_summary = pd.DataFrame({
    "Missing Count"  : missing_counts,
    "Missing Percent": missing_percent.round(2)
})
# Only show columns with at least one missing value
missing_with_values = missing_summary[missing_summary["Missing Count"] > 0]
if missing_with_values.empty:
    print("   ✅ No missing values found!")
else:
    print(missing_with_values.to_string())
    print(f"\n   Total missing cells: {heart_df.isnull().sum().sum()}")

print("\n" + "=" * 65)
print("✅ Phase 1 Complete — Dataset Successfully Loaded!")
print("=" * 65)
print("""
📌 Summary of what we accomplished in Phase 1:
   1. Verified the dataset file exists at the expected path.
   2. Defined 14 meaningful column names from UCI documentation.
   3. Loaded the raw CSV, treating '?' as NaN automatically.
   4. Binarized the Target: 0 = Healthy, 1 = Heart Disease.
   5. Inspected the first and last 10 rows for a sanity check.
   6. Confirmed shape: 303 patients × 14 columns.
   7. Examined data types — all numeric (no 'object' surprises).
   8. Reviewed df.info() to spot non-null counts per column.
   9. Reviewed df.describe() for statistical summaries.
  10. Identified which columns have missing values (CA and Thal).

🔜 Ready for Phase 2: Data Cleaning — awaiting your confirmation.
""")
