# =============================================================================
# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
# =============================================================================
# Phase 3: Descriptive Statistics & Exploratory Data Analysis (EDA)
# Author   : Senior Data Scientist & Python Mentor
# Date     : November 2026
# Module   : notebooks/phase3_descriptive_stats.py
# =============================================================================

# -----------------------------------------------------------------------------
# WHY DO WE IMPORT THESE LIBRARIES?
# - os: For reliable file path resolution.
# - pandas: For calculating summary statistics, percentiles, and correlation.
# - numpy: For numerical matrix computations.
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

# Set display options for comprehensive table rendering
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.2f}".format)

# =============================================================================
# STEP 1: LOAD CLEANED DATASET (From Phase 2)
# =============================================================================
# WHY: We always base statistical analysis on the verified, clean dataset
# produced in Phase 2 to guarantee zero NaN values and proper data types.
# =============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(project_root, "dataset", "cleaned_cardiopredict.csv")

print("=" * 75)
print("   CardioPredict — Phase 3: Comprehensive Descriptive Statistics & EDA")
print("=" * 75)

if not os.path.exists(cleaned_path):
    raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}. Run Phase 2 first.")

heart_df = pd.read_csv(cleaned_path)
print(f"✅ Cleaned dataset loaded: {heart_df.shape[0]} patients, {heart_df.shape[1]} attributes.")

# Continuous clinical features for in-depth parametric & non-parametric evaluation
continuous_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# =============================================================================
# STEP 2: MEASURES OF CENTRAL TENDENCY (Mean, Median, Mode)
# =============================================================================
# WHY: Central tendency tells us where the "center" or typical value of the
# distribution lies. Comparing Mean vs. Median reveals skewness and outlier
# influence (Mean is sensitive to extreme values, Median is robust).
# =============================================================================
print("\n" + "=" * 75)
print("📍 STEP 2: Measures of Central Tendency (Mean, Median, Mode)")
print("=" * 75)

central_tendency = pd.DataFrame({
    "Mean": [heart_df[col].mean() for col in continuous_features],
    "Median": [heart_df[col].median() for col in continuous_features],
    "Mode": [heart_df[col].mode()[0] for col in continuous_features]
}, index=continuous_features)

print(central_tendency.to_string())

# =============================================================================
# STEP 3: MEASURES OF DISPERSION & SPREAD (Min, Max, Range, Variance, Std Dev)
# =============================================================================
# WHY: Central tendency alone is insufficient. We must quantify the variability,
# risk boundaries, and spread of clinical measurements.
# =============================================================================
print("\n" + "=" * 75)
print("📏 STEP 3: Measures of Dispersion (Min, Max, Range, Variance, Std Dev)")
print("=" * 75)

dispersion_df = pd.DataFrame({
    "Min": [heart_df[col].min() for col in continuous_features],
    "Max": [heart_df[col].max() for col in continuous_features],
    "Range": [heart_df[col].max() - heart_df[col].min() for col in continuous_features],
    "Variance": [heart_df[col].var() for col in continuous_features],
    "Std Dev": [heart_df[col].std() for col in continuous_features]
}, index=continuous_features)

print(dispersion_df.to_string())

# =============================================================================
# STEP 4: QUARTILES & INTERQUARTILE RANGE (Q1, Q2, Q3, IQR)
# =============================================================================
# WHY: Quartiles partition the sorted data into 4 equal segments.
# IQR (Q3 - Q1) represents the middle 50% of the patient cohort and serves as
# the foundation for identifying statistical outliers via Tukey's fences.
# =============================================================================
print("\n" + "=" * 75)
print("📊 STEP 4: Quartiles & Interquartile Range (IQR)")
print("=" * 75)

q1_series = heart_df[continuous_features].quantile(0.25)
q2_series = heart_df[continuous_features].quantile(0.50)
q3_series = heart_df[continuous_features].quantile(0.75)
iqr_series = q3_series - q1_series

quartiles_df = pd.DataFrame({
    "Q1 (25th %)": q1_series,
    "Q2 (Median)": q2_series,
    "Q3 (75th %)": q3_series,
    "IQR": iqr_series,
    "Lower Fence (Q1 - 1.5*IQR)": q1_series - 1.5 * iqr_series,
    "Upper Fence (Q3 + 1.5*IQR)": q3_series + 1.5 * iqr_series
})

print(quartiles_df.to_string())

# =============================================================================
# STEP 5: MEASURES OF SHAPE (Skewness & Kurtosis)
# =============================================================================
# WHY:
# - Skewness measures asymmetry about the mean.
#   * S = 0: Symmetrical (Gaussian)
#   * S > 0: Positive/Right-skewed (tail points right)
#   * S < 0: Negative/Left-skewed (tail points left)
# - Kurtosis measures tail heaviness (tailedness) relative to a normal curve.
#   * Excess Kurtosis = 0: Mesokurtic (Normal)
#   * Excess Kurtosis > 0: Leptokurtic (Heavy tails, higher outlier risk)
#   * Excess Kurtosis < 0: Platykurtic (Light tails, flat distribution)
# =============================================================================
print("\n" + "=" * 75)
print("📐 STEP 5: Distribution Shape (Skewness & Kurtosis)")
print("=" * 75)

shape_df = pd.DataFrame({
    "Skewness": [heart_df[col].skew() for col in continuous_features],
    "Skewness Interpretation": [
        "Approx. Symmetric" if abs(heart_df[col].skew()) < 0.5
        else "Moderately Right-Skewed" if heart_df[col].skew() > 0
        else "Moderately Left-Skewed"
        for col in continuous_features
    ],
    "Kurtosis": [heart_df[col].kurtosis() for col in continuous_features],
    "Kurtosis Interpretation": [
        "Leptokurtic (Heavy tails)" if heart_df[col].kurtosis() > 0.5
        else "Platykurtic (Flat/Light tails)" if heart_df[col].kurtosis() < -0.5
        else "Mesokurtic (Near Normal)"
        for col in continuous_features
    ]
}, index=continuous_features)

print(shape_df.to_string())

# =============================================================================
# STEP 6: COMPREHENSIVE STATISTICAL SUMMARY TABLE
# =============================================================================
# Combining all 12 key descriptive metrics into a unified presentation table.
# =============================================================================
print("\n" + "=" * 75)
print("📋 STEP 6: Master Descriptive Statistics Summary Table")
print("=" * 75)

summary_master = pd.DataFrame({
    "Mean": central_tendency["Mean"],
    "Median": central_tendency["Median"],
    "Mode": central_tendency["Mode"],
    "Min": dispersion_df["Min"],
    "Max": dispersion_df["Max"],
    "Range": dispersion_df["Range"],
    "Variance": dispersion_df["Variance"],
    "Std Dev": dispersion_df["Std Dev"],
    "Q1": quartiles_df["Q1 (25th %)"],
    "Q3": quartiles_df["Q3 (75th %)"],
    "IQR": quartiles_df["IQR"],
    "Skewness": shape_df["Skewness"],
    "Kurtosis": shape_df["Kurtosis"]
})

print(summary_master.to_string())

# =============================================================================
# STEP 7: CORRELATION MATRIX & TARGET ASSOCIATIONS
# =============================================================================
# WHY: Pearson correlation coefficient r measures linear dependency between
# features. Ranges from -1.0 (perfect negative) to +1.0 (perfect positive).
# Correlation with Target exposes the most diagnostically potent features.
# =============================================================================
print("\n" + "=" * 75)
print("🔗 STEP 7: Full Pearson Correlation Matrix & Target Correlation Ranking")
print("=" * 75)

full_correlation_matrix = heart_df.corr(method="pearson")

print("Correlation of All Features with Target (Sorted by Strength):")
target_correlation = full_correlation_matrix["Target"].sort_values(ascending=False)
target_corr_df = pd.DataFrame({
    "Pearson r with Target": target_correlation,
    "Relationship Direction": [
        "Positive (Increases Risk)" if r > 0 else "Negative (Protective / Decreases)"
        for r in target_correlation
    ],
    "Strength": [
        "Perfect" if abs(r) == 1.0
        else "Strong" if abs(r) >= 0.5
        else "Moderate" if abs(r) >= 0.3
        else "Weak" if abs(r) >= 0.1
        else "Negligible"
        for r in target_correlation
    ]
})
print(target_corr_df.to_string())

print("\n" + "=" * 75)
print("✅ Phase 3 Complete — Descriptive Statistics & EDA Executed!")
print("=" * 75)
