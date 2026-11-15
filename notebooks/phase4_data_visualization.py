# =============================================================================
# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
# =============================================================================
# Phase 4: Data Visualization & Exploratory Graphical Analysis
# Author   : Senior Data Scientist & Python Mentor
# Date     : November 2026
# Module   : notebooks/phase4_data_visualization.py
# =============================================================================

# -----------------------------------------------------------------------------
# WHY DO WE CONFIGURE MATPLOTLIB WITH 'Agg' BACKEND?
# In automated data science scripts and server environments, a GUI window
# is not available. Using the non-interactive 'Agg' backend ensures plots
# render headlessly and save to disk cleanly without blocking execution.
# -----------------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Global visual aesthetics configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_dataset_path = os.path.join(project_root, "dataset", "cleaned_cardiopredict.csv")
images_dir = os.path.join(project_root, "static", "images")
os.makedirs(images_dir, exist_ok=True)

print("=" * 75)
print("   CardioPredict — Phase 4: Clinical Data Visualizations")
print("=" * 75)

# Load cleaned dataset
heart_df = pd.read_csv(cleaned_dataset_path)
print(f"✅ Loaded cleaned dataset: {heart_df.shape[0]} patients, {heart_df.shape[1]} features.")
print(f"📁 Destination directory for plots: {images_dir}\n")

# Color palette definition for clinical clarity
palette_colors = ["#2ecc71", "#e74c3c"]  # Emerald Green (Healthy=0) vs Crimson Red (Disease=1)
sns.set_palette(sns.color_palette(palette_colors))

# =============================================================================
# PLOT 1: HISTOGRAM (Age Distribution with KDE)
# =============================================================================
print("📊 Generating Plot 1: Histogram (Age Distribution)...")
fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=heart_df,
    x="Age",
    hue="Target",
    bins=16,
    kde=True,
    palette=palette_colors,
    multiple="stack",
    edgecolor="white",
    alpha=0.85,
    ax=ax
)

ax.set_title("1. Age Distribution Stratified by Heart Disease Status", fontweight="bold", pad=15)
ax.set_xlabel("Patient Age (Years)")
ax.set_ylabel("Patient Count")
ax.legend(title="Diagnosis", labels=["Heart Disease", "Healthy"])
plt.tight_layout()
plot1_path = os.path.join(images_dir, "01_histogram_age.png")
fig.savefig(plot1_path)
plt.close(fig)
print(f"   Saved: {plot1_path}")

# =============================================================================
# PLOT 2: COUNT PLOT (Sex vs Heart Disease)
# =============================================================================
print("📊 Generating Plot 2: Count Plot (Gender vs Disease Diagnosis)...")
fig, ax = plt.subplots(figsize=(8, 6))

plot_data = heart_df.copy()
plot_data["Sex_Label"] = plot_data["Sex"].map({0: "Female", 1: "Male"})
plot_data["Target_Label"] = plot_data["Target"].map({0: "Healthy", 1: "Heart Disease"})

sns.countplot(
    data=plot_data,
    x="Sex_Label",
    hue="Target_Label",
    palette=palette_colors,
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")

ax.set_title("2. Patient Frequency by Gender and Diagnosis Status", fontweight="bold", pad=15)
ax.set_xlabel("Biological Sex")
ax.set_ylabel("Number of Patients")
ax.legend(title="Clinical Diagnosis")
plt.tight_layout()
plot2_path = os.path.join(images_dir, "02_countplot_gender.png")
fig.savefig(plot2_path)
plt.close(fig)
print(f"   Saved: {plot2_path}")

# =============================================================================
# PLOT 3: BAR CHART (Heart Disease Prevalence by Chest Pain Type)
# =============================================================================
print("📊 Generating Plot 3: Bar Chart (Prevalence by Chest Pain Type)...")
fig, ax = plt.subplots(figsize=(9, 6))

cp_labels = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-Anginal Pain",
    4: "Asymptomatic"
}
cp_prevalence = heart_df.groupby("ChestPainType")["Target"].agg(
    Total="count",
    Diseased="sum"
).reset_index()
cp_prevalence["Prevalence_Pct"] = (cp_prevalence["Diseased"] / cp_prevalence["Total"]) * 100
cp_prevalence["ChestPain_Name"] = cp_prevalence["ChestPainType"].map(cp_labels)

bars = ax.bar(
    cp_prevalence["ChestPain_Name"],
    cp_prevalence["Prevalence_Pct"],
    color=["#3498db", "#9b59b6", "#f39c12", "#e74c3c"],
    edgecolor="black",
    width=0.55
)

for bar, total, diseased in zip(bars, cp_prevalence["Total"], cp_prevalence["Diseased"]):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5,
            f"{yval:.1f}%\n({diseased}/{total})",
            ha="center", va="bottom", fontweight="bold", fontsize=10)

ax.set_ylim(0, 100)
ax.set_title("3. Heart Disease Prevalence Across Chest Pain Types", fontweight="bold", pad=15)
ax.set_xlabel("Clinical Chest Pain Classification")
ax.set_ylabel("Disease Prevalence (%)")
plt.tight_layout()
plot3_path = os.path.join(images_dir, "03_barchart_chestpain.png")
fig.savefig(plot3_path)
plt.close(fig)
print(f"   Saved: {plot3_path}")

# =============================================================================
# PLOT 4: PIE CHART (Target Class Balance)
# =============================================================================
print("📊 Generating Plot 4: Pie Chart (Overall Cohort Diagnosis Split)...")
fig, ax = plt.subplots(figsize=(7, 7))

target_counts = heart_df["Target"].value_counts()
labels = [f"Healthy\n(n = {target_counts[0]})", f"Heart Disease\n(n = {target_counts[1]})"]
colors = ["#2ecc71", "#e74c3c"]
explode = (0.05, 0.0)

wedges, texts, autotexts = ax.pie(
    target_counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    explode=explode,
    shadow=True,
    textprops={"fontsize": 12, "fontweight": "bold"}
)

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(13)

ax.set_title("4. CardioPredict Target Class Distribution", fontweight="bold", pad=15)
plt.tight_layout()
plot4_path = os.path.join(images_dir, "04_piechart_target.png")
fig.savefig(plot4_path)
plt.close(fig)
print(f"   Saved: {plot4_path}")

# =============================================================================
# PLOT 5: BOX PLOT (RestingBP, Cholesterol, MaxHR Stratified by Diagnosis)
# =============================================================================
print("📊 Generating Plot 5: Box Plot (Clinical Continuous Features)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

features_to_plot = [
    ("RestingBP", "Resting Blood Pressure (mm Hg)", axes[0]),
    ("Cholesterol", "Serum Cholesterol (mg/dl)", axes[1]),
    ("MaxHR", "Maximum Heart Rate Achieved (bpm)", axes[2])
]

for feature, label, ax in features_to_plot:
    sns.boxplot(
        data=heart_df,
        x="Target",
        y=feature,
        hue="Target",
        legend=False,
        palette=palette_colors,
        width=0.45,
        fliersize=5,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy (0)", "Disease (1)"])
    ax.set_title(f"{label.split('(')[0].strip()}", fontweight="bold")
    ax.set_xlabel("Clinical Diagnosis")
    ax.set_ylabel(label)

fig.suptitle("5. Box Plot Distributions & Outliers by Target Diagnosis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plot5_path = os.path.join(images_dir, "05_boxplot_clinical_features.png")
fig.savefig(plot5_path, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {plot5_path}")

# =============================================================================
# PLOT 6: VIOLIN PLOT (Max Heart Rate by Heart Disease Status)
# =============================================================================
print("📊 Generating Plot 6: Violin Plot (MaxHR Density & Quartiles)...")
fig, ax = plt.subplots(figsize=(9, 6))

sns.violinplot(
    data=heart_df,
    x="Target",
    y="MaxHR",
    hue="Target",
    legend=False,
    palette=palette_colors,
    inner="quartile",
    cut=0,
    ax=ax
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Healthy (0)", "Heart Disease (1)"])
ax.set_title("6. Violin Plot: Max Heart Rate Density and Quartiles by Diagnosis", fontweight="bold", pad=15)
ax.set_xlabel("Diagnosis Status")
ax.set_ylabel("Max Heart Rate (bpm)")
plt.tight_layout()
plot6_path = os.path.join(images_dir, "06_violinplot_maxhr.png")
fig.savefig(plot6_path)
plt.close(fig)
print(f"   Saved: {plot6_path}")

# =============================================================================
# PLOT 7: SCATTER PLOT (Age vs Max Heart Rate with Linear Regression Trends)
# =============================================================================
print("📊 Generating Plot 7: Scatter Plot (Age vs MaxHR with Regression Lines)...")
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=heart_df,
    x="Age",
    y="MaxHR",
    hue="Target",
    palette=palette_colors,
    alpha=0.8,
    s=70,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

# Linear trend lines
sns.regplot(
    data=heart_df[heart_df["Target"] == 0],
    x="Age", y="MaxHR",
    scatter=False, ax=ax,
    color="#2ecc71"
)
sns.regplot(
    data=heart_df[heart_df["Target"] == 1],
    x="Age", y="MaxHR",
    scatter=False, ax=ax,
    color="#e74c3c"
)

ax.set_title("7. Age vs. Maximum Heart Rate Achieved (Chronotropic Gradient)", fontweight="bold", pad=15)
ax.set_xlabel("Patient Age (Years)")
ax.set_ylabel("Maximum Heart Rate Achieved (bpm)")
ax.legend(title="Cohort", labels=["Healthy (0)", "Disease (1)", "Healthy Trend", "Disease Trend"])
plt.tight_layout()
plot7_path = os.path.join(images_dir, "07_scatterplot_age_maxhr.png")
fig.savefig(plot7_path)
plt.close(fig)
print(f"   Saved: {plot7_path}")

# =============================================================================
# PLOT 8: PAIR PLOT (Multi-variable Pairwise Feature Distributions)
# =============================================================================
print("📊 Generating Plot 8: Pair Plot (Key Continuous Features Matrix)...")
pairplot_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Target"]
pairplot_data = heart_df[pairplot_features].copy()

pair_grid = sns.pairplot(
    pairplot_data,
    hue="Target",
    palette=palette_colors,
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.65, "s": 35, "edgecolor": "none"}
)
pair_grid.fig.suptitle("8. Pairwise Feature Interaction Matrix Stratified by Diagnosis", y=1.02, fontweight="bold", fontsize=14)
plot8_path = os.path.join(images_dir, "08_pairplot_multivariate.png")
pair_grid.savefig(plot8_path, bbox_inches="tight")
plt.close(pair_grid.fig)
print(f"   Saved: {plot8_path}")

# =============================================================================
# PLOT 9: CORRELATION HEATMAP (Full Annotated Matrix)
# =============================================================================
print("📊 Generating Plot 9: Correlation Heatmap (Full 14x14 Matrix)...")
fig, ax = plt.subplots(figsize=(12, 10))

corr_matrix = heart_df.corr(method="pearson")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.6,
    vmax=0.6,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
    ax=ax
)

ax.set_title("9. Full Pearson Correlation Matrix of All 14 Clinical Features", fontweight="bold", pad=15)
plt.tight_layout()
plot9_path = os.path.join(images_dir, "09_correlation_heatmap.png")
fig.savefig(plot9_path)
plt.close(fig)
print(f"   Saved: {plot9_path}")

# =============================================================================
# PLOT 10: DISTRIBUTION PLOT (Oldpeak ST Depression KDE Comparison)
# =============================================================================
print("📊 Generating Plot 10: Distribution Plot (Oldpeak ST Depression KDE)...")
fig, ax = plt.subplots(figsize=(10, 6))

sns.kdeplot(
    data=heart_df[heart_df["Target"] == 0]["Oldpeak"],
    label="Healthy (Target = 0)",
    color="#2ecc71",
    fill=True,
    alpha=0.45,
    linewidth=2,
    ax=ax
)
sns.kdeplot(
    data=heart_df[heart_df["Target"] == 1]["Oldpeak"],
    label="Heart Disease (Target = 1)",
    color="#e74c3c",
    fill=True,
    alpha=0.45,
    linewidth=2,
    ax=ax
)

ax.set_title("10. Probability Density of ST Depression (Oldpeak) by Diagnosis", fontweight="bold", pad=15)
ax.set_xlabel("ST Depression Induced by Exercise Relative to Rest (mm)")
ax.set_ylabel("Probability Density")
ax.legend(title="Clinical Diagnosis")
plt.tight_layout()
plot10_path = os.path.join(images_dir, "10_distribution_oldpeak.png")
fig.savefig(plot10_path)
plt.close(fig)
print(f"   Saved: {plot10_path}")

print("\n" + "=" * 75)
print("✅ Phase 4 Complete — All 10 Diagnostic Visualizations Successfully Generated!")
print("=" * 75)
