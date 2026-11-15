# =============================================================================
# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
# =============================================================================
# Phase 5: Inferential Statistics & Hypothesis Testing
# Author   : Senior Data Scientist & Python Mentor
# Date     : November 2026
# Module   : notebooks/phase5_inferential_statistics.py
# =============================================================================

# -----------------------------------------------------------------------------
# WHY DO WE USE SCIPY.STATS?
# SciPy (Scientific Python) is the foundational scientific library in Python
# for inferential statistics. It provides rigorous implementations of parametric
# (t-test, ANOVA) and non-parametric / categorical tests (Chi-Square) with
# exact p-value computations and hypothesis testing frameworks.
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats

# Clean terminal output configuration
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)

# Resolve project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_dataset_path = os.path.join(project_root, "dataset", "cleaned_cardiopredict.csv")

print("=" * 75)
print("   CardioPredict — Phase 5: Inferential Statistics & Hypothesis Testing")
print("=" * 75)

# Load cleaned dataset
heart_df = pd.read_csv(cleaned_dataset_path)
print(f"✅ Loaded cleaned dataset: {heart_df.shape[0]} patient records, {heart_df.shape[1]} features.\n")

# Significance threshold (alpha)
ALPHA = 0.05

# =============================================================================
# 1. INDEPENDENT TWO-SAMPLE T-TEST
# =============================================================================
# GOAL: Compare serum cholesterol levels between patients WITH heart disease
# (Target = 1) and WITHOUT heart disease (Target = 0).
# =============================================================================
print("=" * 75)
print("1. INDEPENDENT TWO-SAMPLE T-TEST: Cholesterol vs. Heart Disease")
print("=" * 75)

chol_healthy = heart_df[heart_df["Target"] == 0]["Cholesterol"]
chol_diseased = heart_df[heart_df["Target"] == 1]["Cholesterol"]

# Step 1.1: Descriptive metrics per group
n_healthy, mean_healthy, std_healthy = len(chol_healthy), chol_healthy.mean(), chol_healthy.std()
n_diseased, mean_diseased, std_diseased = len(chol_diseased), chol_diseased.mean(), chol_diseased.std()

print(f"Group 1 - Healthy Patients    (Target = 0): n = {n_healthy:3d}, Mean = {mean_healthy:.2f} mg/dl, Std = {std_healthy:.2f}")
print(f"Group 2 - Diseased Patients   (Target = 1): n = {n_diseased:3d}, Mean = {mean_diseased:.2f} mg/dl, Std = {std_diseased:.2f}")

# Step 1.2: Test for Homogeneity of Variances (Levene's Test)
levene_stat, levene_p = stats.levene(chol_healthy, chol_diseased)
print(f"\nLevene's Variance Test: W = {levene_stat:.4f}, p-value = {levene_p:.4f}")
equal_variances = levene_p > ALPHA
print(f"Homoscedasticity assumption: {'MET (Equal variances assumed)' if equal_variances else 'VIOLATED (Welch t-test required)'}")

# Step 1.3: Compute Student's Two-Sample t-test
t_statistic, t_pvalue = stats.ttest_ind(chol_healthy, chol_diseased, equal_var=equal_variances)

print(f"\nT-Test Results:")
print(f"   Test Statistic (t) : {t_statistic:.4f}")
print(f"   Degrees of Freedom : {n_healthy + n_diseased - 2}")
print(f"   P-value            : {t_pvalue:.4f}")
print(f"   Significance Level : alpha = {ALPHA}")

if t_pvalue < ALPHA:
    print("   Decision: REJECT Null Hypothesis (H0). Statistically significant difference exists.")
else:
    print("   Decision: FAIL TO REJECT Null Hypothesis (H0). No statistically significant difference.")

# =============================================================================
# 2. CHI-SQUARE TEST OF INDEPENDENCE
# =============================================================================
# GOAL: Determine whether Biological Sex is significantly associated with
# Heart Disease diagnosis.
# =============================================================================
print("\n" + "=" * 75)
print("2. CHI-SQUARE TEST OF INDEPENDENCE: Gender vs. Heart Disease")
print("=" * 75)

# Step 2.1: Construct Contingency Table (Cross-tabulation)
contingency_table = pd.crosstab(
    heart_df["Sex"],
    heart_df["Target"],
    margins=True,
    margins_name="Total"
)
contingency_table.index = ["Female (0)", "Male (1)", "Total"]
contingency_table.columns = ["Healthy (0)", "Heart Disease (1)", "Total"]

print("Observed Contingency Table (Cross-Tabulation):")
print(contingency_table.to_string())

# Step 2.2: Compute Chi-Square Statistic, p-value, DoF, and Expected Frequencies
# Using the 2x2 observed values without margins
observed_table = pd.crosstab(heart_df["Sex"], heart_df["Target"])
chi2_stat, chi2_pvalue, dof, expected_frequencies = stats.chi2_contingency(observed_table)

expected_df = pd.DataFrame(
    expected_frequencies,
    index=["Female (0)", "Male (1)"],
    columns=["Healthy (0)", "Heart Disease (1)"]
)

print("\nExpected Frequencies Table (Under H0 of Independence):")
print(expected_df.round(2).to_string())

print(f"\nChi-Square Test Results:")
print(f"   Chi-Square Statistic (χ²) : {chi2_stat:.4f}")
print(f"   Degrees of Freedom (df)   : {dof}  (calculated as (r - 1) * (c - 1) = (2-1)*(2-1) = 1)")
print(f"   P-value                   : {chi2_pvalue:.4e}")
print(f"   Significance Level        : alpha = {ALPHA}")

if chi2_pvalue < ALPHA:
    print("   Decision: REJECT Null Hypothesis (H0). Significant association between Gender and Disease.")
else:
    print("   Decision: FAIL TO REJECT Null Hypothesis (H0). Gender and Disease are independent.")

# =============================================================================
# 3. ONE-WAY ANALYSIS OF VARIANCE (ANOVA)
# =============================================================================
# GOAL: Compare mean serum cholesterol levels across the four distinct
# Chest Pain categories (Typical, Atypical, Non-Anginal, Asymptomatic).
# =============================================================================
print("\n" + "=" * 75)
print("3. ONE-WAY ANOVA: Serum Cholesterol across Chest Pain Types")
print("=" * 75)

# Group cholesterol by ChestPainType
cp1_chol = heart_df[heart_df["ChestPainType"] == 1]["Cholesterol"]
cp2_chol = heart_df[heart_df["ChestPainType"] == 2]["Cholesterol"]
cp3_chol = heart_df[heart_df["ChestPainType"] == 3]["Cholesterol"]
cp4_chol = heart_df[heart_df["ChestPainType"] == 4]["Cholesterol"]

cp_groups = [
    ("Type 1: Typical Angina", cp1_chol),
    ("Type 2: Atypical Angina", cp2_chol),
    ("Type 3: Non-Anginal Pain", cp3_chol),
    ("Type 4: Asymptomatic", cp4_chol)
]

print("Descriptive Statistics of Cholesterol by Chest Pain Category:")
for name, grp in cp_groups:
    print(f"   {name:<25}: n = {len(grp):3d}, Mean = {grp.mean():.2f} mg/dl, Std = {grp.std():.2f}")

# Compute One-Way ANOVA F-test
f_statistic, anova_pvalue = stats.f_oneway(cp1_chol, cp2_chol, cp3_chol, cp4_chol)

print(f"\nOne-Way ANOVA Results:")
print(f"   F-Statistic        : {f_statistic:.4f}")
print(f"   Between Groups df  : {len(cp_groups) - 1}")
print(f"   Within Groups df   : {len(heart_df) - len(cp_groups)}")
print(f"   P-value            : {anova_pvalue:.4f}")
print(f"   Significance Level : alpha = {ALPHA}")

if anova_pvalue < ALPHA:
    print("   Decision: REJECT Null Hypothesis (H0). At least one group mean is significantly different.")
else:
    print("   Decision: FAIL TO REJECT Null Hypothesis (H0). No significant difference across groups.")

# =============================================================================
# 4. PEARSON CORRELATION ANALYSIS WITH SIGNIFICANCE TESTING
# =============================================================================
# GOAL: Calculate Pearson Correlation Coefficient (r) and exact two-tailed
# p-values for all important continuous and discrete numerical features.
# =============================================================================
print("\n" + "=" * 75)
print("4. PEARSON CORRELATION ANALYSIS (Linear Association & Significance)")
print("=" * 75)

key_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "CA"]
correlation_results = []

for feature in key_features:
    r_val, p_val = stats.pearsonr(heart_df[feature], heart_df["Target"])
    
    # Classify direction
    direction = "Positive" if r_val > 0 else "Negative"
    
    # Classify strength
    abs_r = abs(r_val)
    if abs_r >= 0.50:
        strength = "Strong"
    elif abs_r >= 0.30:
        strength = "Moderate"
    elif abs_r >= 0.10:
        strength = "Weak"
    else:
        strength = "Negligible"
        
    # Statistical significance at alpha = 0.05
    is_significant = p_val < ALPHA
    
    correlation_results.append({
        "Feature": feature,
        "Pearson r": r_val,
        "P-value": p_val,
        "Direction": direction,
        "Strength": strength,
        "Statistically Significant": "Yes (p < 0.05)" if is_significant else "No (p >= 0.05)"
    })

corr_summary_df = pd.DataFrame(correlation_results)
print(corr_summary_df.to_string(index=False))

print("\n" + "=" * 75)
print("✅ Phase 5 Complete — All 4 Inferential Statistical Tests Executed!")
print("=" * 75)
