# CardioPredict: Heart Disease Prediction & Analytics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Production--Ready-success.svg)](#)

> **CardioPredict** is an end-to-end clinical decision support system and healthcare analytics web platform developed for an academic Introduction to Data Science (IDS) capstone project. By training and benchmarking multiple supervised machine learning algorithms on clinical biomarker data from the renowned Cleveland Heart Disease database, CardioPredict delivers instantaneous, high-precision ischemic heart disease risk stratification with **90.16% test accuracy**, **92.86% clinical sensitivity (recall)**, and an **ROC-AUC of 0.9654**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [End-to-End Pipeline Architecture](#end-to-end-pipeline-architecture)
4. [Project Folder Structure](#project-folder-structure)
5. [Dataset Information](#dataset-information)
6. [Machine Learning Models & Benchmark](#machine-learning-models--benchmark)
7. [Evaluation Metrics & Rationale](#evaluation-metrics--rationale)
8. [Technology Stack](#technology-stack)
9. [Installation & Setup](#installation--setup)
10. [Running the Web Application](#running-the-web-application)
11. [Screenshots & UI Walkthrough](#screenshots--ui-walkthrough)
12. [Future Enhancements](#future-enhancements)
13. [Project Conclusion](#conclusion)
14. [License](#license)

---

## 1. Project Overview

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, claiming approximately 17.9 million lives each year according to the World Health Organization (WHO). Early detection and non-invasive identification of individuals with coronary artery stenosis or ischemia are critical for early medical intervention, lifestyle modification, and surgical planning.

**CardioPredict** bridges rigorous statistical data science with modern web engineering. Starting from raw clinical data, it follows a strict 10-phase software lifecycle:
* Rigorous median-imputation and data cleaning.
* Parametric and non-parametric exploratory data analysis (EDA).
* Inferential hypothesis testing (Two-Sample t-tests, Pearson correlation, Chi-Square tests of independence, and One-Way ANOVA).
* Scaled feature engineering preserving clinical interpretability.
* Multi-model training and cross-comparative benchmarking across 5 distinct classification paradigms.
* A production-style Flask web application powered by a pre-trained, serialised inference engine with instant risk scoring, probability gauges, and evidence-based clinical recommendations.

---

## 2. Key Features

* **Instant Clinical Risk Stratification**: Computes diagnostic probabilities across 3 clinically grounded risk tiers:
  * **Low Risk / Healthy** ($p < 0.50$): Normal cardiovascular biomarkers.
  * **Moderate Risk** ($0.50 \le p < 0.75$): Borderline ST depressions or elevated resting blood pressure; monitoring advised.
  * **High Risk / Critical** ($p \ge 0.75$): High likelihood of CAD; immediate cardiological evaluation recommended.
* **Zero Retraining Inference**: Web requests consume pre-fitted serialised artifacts (`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`), achieving sub-millisecond inference latency without background model retraining.
* **One-Click Demo Fillers**: Pre-fills the 13-feature intake form with verified Low-Risk and High-Risk patient profiles for rapid testing and academic demonstration.
* **Interactive Healthcare Analytics Dashboard**:
  * Real-time KPI summaries (Test Accuracy, Recall, Precision, ROC-AUC, Specificity).
  * Comprehensive benchmark comparison table across all 5 trained models.
  * Embedded high-resolution exploratory data plots, multi-class confusion matrices, ROC curves, Precision-Recall curves, and feature importance rankings.
* **Tailored Clinical Guidance**: Generates personalized patient recommendations based on resting BP, serum cholesterol, exercise-induced angina, and ST-segment depression.
* **Responsive Healthcare UI**: Crafted with Bootstrap 5, custom medical-blue CSS stylesheets, responsive grids, and subtle UI animations.

---

## 3. End-to-End Pipeline Architecture

```
                                 [ Raw Cleveland Dataset ]
                                            │
                                            ▼
                           Phase 1 & 2: Ingestion & Data Cleaning
                   (Missing values '?' -> Median Imputation -> Binary Target)
                                            │
                                            ▼
                           Phase 3, 4 & 5: EDA & Inferential Statistics
                   (Descriptive stats, 10 plots, t-tests, Chi-Square, ANOVA)
                                            │
                                            ▼
                           Phase 6: Preprocessing & Serialization
                   (Stratified 80/20 split -> StandardScaler -> .pkl export)
                                            │
                                            ▼
                           Phase 7: Machine Learning Model Training
                   (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
                                            │
                                            ▼
                           Phase 8: Benchmarking & Model Evaluation
                   (Confusion Matrices, ROC-AUC 0.9654, PR Curves, Selection)
                                            │
                                            ▼
                           Phase 9: Flask Web Application & Serving
                   (predict.py engine -> app.py controller -> Bootstrap 5 UI)
```

---

## 4. Project Folder Structure

```
CardioPredict/
│
├── app.py                         # Flask web application controller
├── requirements.txt               # Application dependencies manifest
├── README.md                      # Primary project documentation
│
├── dataset/
│   ├── processed.cleveland.data   # Original UCI Cleveland clinical dataset
│   ├── heart_cleaned.csv          # Cleaned dataset (303 rows, 14 columns)
│   └── heart-disease.names        # Original UCI documentation
│
├── pipeline/
│   ├── data_loader.py             # Data ingestion and (X, y) separation
│   ├── preprocessing.py           # Stratified train/test split and StandardScaler
│   ├── feature_engineering.py     # Variable taxonomy and feature selection logic
│   ├── train.py                   # Multi-algorithm model training module
│   ├── evaluate.py                # Benchmarking, metrics, and visualization generator
│   ├── predict.py                 # Production inference engine for web app
│   └── utils.py                   # Path resolution and artifact save/load helpers
│
├── models/
│   ├── best_model.pkl             # Serialised champion model (Random Forest)
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   ├── encoder.pkl                # Categorical encoding mappings
│   ├── feature_columns.pkl        # List of 13 expected input column names
│   ├── logistic_regression.pkl    # Serialised Logistic Regression model
│   ├── decision_tree.pkl          # Serialised Decision Tree model
│   ├── random_forest.pkl          # Serialised Random Forest model
│   ├── svm.pkl                    # Serialised Support Vector Machine model
│   └── knn.pkl                    # Serialised K-Nearest Neighbors model
│
├── evaluation/
│   ├── model_comparison.csv       # Comparative metrics for all 5 models
│   ├── classification_reports.txt # Detailed precision, recall, and f1 text reports
│   └── evaluation_summary.txt     # Executive benchmark summary
│
├── templates/
│   ├── base.html                  # Master layout with navbar, alerts, footer
│   ├── home.html                  # Landing page with hero banner & pipeline cards
│   ├── predict.html               # 13-feature clinical intake form
│   ├── result.html                # Patient risk report & recommendations
│   ├── dashboard.html             # Analytics dashboard with embedded visualizations
│   ├── about.html                 # Dataset origins, methodology & tech stack
│   └── error.html                 # Custom 404 / 500 error display
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom healthcare CSS stylesheet
│   ├── js/
│   │   └── script.js              # Client-side form helpers & demo fillers
│   └── images/                    # Visualizations generated during Phases 4 & 8
│       ├── 01_histogram_age.png
│       ├── ...
│       ├── 10_distribution_oldpeak.png
│       ├── confusion_matrix_*.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       ├── model_accuracy_comparison.png
│       └── feature_importance.png
│
├── documentation/
│   ├── Project_Report.md          # Comprehensive academic project report
│   ├── Viva_Questions.md          # 50 technical viva defense Q&A
│   ├── Requirements.md            # Hardware & software requirements
│   ├── User_Guide.md              # Installation & operational manual
│   ├── Project_Structure.md       # Exhaustive file-by-file documentation
│   └── Final_Summary.md           # End-to-end 10-phase summary
│
└── notebooks/
    └── CardioPredict_EDA.ipynb    # Interactive Jupyter notebook covering Phases 1–8
```

---

## 5. Dataset Information

The project utilizes the **Cleveland Clinic Foundation Heart Disease Dataset** hosted on the UCI Machine Learning Repository, collected by Dr. William Steinbrunn, Dr. Matthias Pfisterer, and Dr. Robert Detrano.

* **Total Samples**: 303 patient records
* **Clean Samples**: 303 rows (6 missing values imputed using robust feature medians)
* **Number of Attributes**: 13 input features + 1 target variable

### Feature Dictionary

| # | Attribute | Type | Scale / Range | Description |
|---|---|---|---|---|
| 1 | `Age` | Numerical | 29 – 77 yrs | Age of the patient in years |
| 2 | `Sex` | Binary | 1 = Male, 0 = Female | Biological sex |
| 3 | `ChestPainType` | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic | Nature of chest pain reported |
| 4 | `RestingBP` | Numerical | 94 – 200 mm Hg | Resting systolic blood pressure on admission |
| 5 | `Cholesterol` | Numerical | 126 – 564 mg/dL | Serum cholesterol level |
| 6 | `FastingBS` | Binary | 1 = > 120 mg/dL, 0 = Normal | Fasting blood sugar level |
| 7 | `RestingECG` | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy | Resting electrocardiographic findings |
| 8 | `MaxHR` | Numerical | 71 – 202 bpm | Maximum heart rate achieved during exercise stress test |
| 9 | `ExerciseAngina` | Binary | 1 = Yes, 0 = No | Exercise-induced angina pectoris |
| 10 | `Oldpeak` | Numerical | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest |
| 11 | `ST_Slope` | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping | Peak exercise ST segment slope |
| 12 | `CA` | Numerical | 0, 1, 2, 3 vessels | Number of major coronary vessels colored by fluoroscopy |
| 13 | `Thal` | Categorical | 3: Normal, 6: Fixed defect, 7: Reversible defect | Thallium scintigraphy stress test result |
| 14 | `Target` | Binary | 0 = Healthy, 1 = Heart Disease | Target diagnosis ($0 \to 0$, $1..4 \to 1$) |

---

## 6. Machine Learning Models & Benchmark

All models were evaluated using an **80/20 Stratified Train-Test Split** ($N_{\text{train}} = 242$, $N_{\text{test}} = 61$) to preserve the exact class ratio (54.1% healthy, 45.9% disease).

| Algorithm | Training Accuracy | Testing Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **100.00%** | **90.16%** | **86.67%** | **92.86%** | **89.66%** | **0.9654** | **87.88%** |
| **Logistic Regression** | 85.54% | 85.25% | 82.76% | 85.71% | 84.21% | 0.9080 | 84.85% |
| **Support Vector Machine (RBF)** | 91.32% | 86.89% | 83.33% | 89.29% | 86.21% | 0.9232 | 84.85% |
| **K-Nearest Neighbors ($k=5$)** | 87.19% | 81.97% | 79.31% | 82.14% | 80.70% | 0.8874 | 81.82% |
| **Decision Tree** | 100.00% | 77.05% | 74.07% | 71.43% | 72.73% | 0.7662 | 81.82% |

### Champion Model Selection Rationale
**Random Forest Classifier** was selected as the production model because:
1. **Highest Sensitivity / Recall (92.86%)**: In clinical diagnostics, **False Negatives** (failing to identify an afflicted patient) are catastrophic. Random Forest missed only 2 diseased cases out of 28 in the holdout test set.
2. **Superior Discriminative Power**: An ROC-AUC of **0.9654** indicates near-optimal discrimination across all decision thresholds.
3. **Resilience to Overfitting**: By aggregating 100 decorrelated decision trees, Random Forest avoids the high-variance pitfall of individual Decision Trees (77.05% test accuracy).

---

## 7. Evaluation Metrics & Rationale

* **Recall / Clinical Sensitivity** ($\frac{TP}{TP + FN}$): Critical to minimize false negatives; a life-threatening cardiac event must never be diagnosed as healthy.
* **Precision** ($\frac{TP}{TP + FP}$): Minimizes unnecessary invasive angiograms and anxiety caused by false positives.
* **F1-Score**: Harmonic mean of Precision and Recall, proving balanced performance.
* **Specificity** ($\frac{TN}{TN + FP}$): True negative rate, confirming healthy patients are correctly flagged.
* **ROC-AUC**: Evaluates model discrimination across all classification thresholds independent of the arbitrary 0.5 cutoff.

---

## 8. Technology Stack

* **Backend & ML Pipeline**: Python 3.10+, Pandas, NumPy, Scikit-Learn, SciPy, Joblib
* **Web Framework**: Flask 3.1+ (Jinja2 templating, RESTful routing, session flash messaging)
* **Frontend UI**: HTML5, CSS3, JavaScript, Bootstrap 5.3, Bootstrap Icons
* **Data Visualization**: Matplotlib 3.11+, Seaborn 0.13+
* **Environment & Tools**: PowerShell / Bash, VS Code, Git

---

## 9. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13 installed.
* Git installed on your system.

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/CardioPredict.git
   cd CardioPredict
   ```

2. **Create a Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade Pip & Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 10. Running the Web Application

With the virtual environment active, launch the Flask server:

```bash
python app.py
```

Expected terminal output:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 11. Screenshots & UI Walkthrough

*(Placeholder anchors for submission screenshots)*

| Section | Preview |
|---|---|
| **1. Landing Page (`/`)** | ![Home Page](documentation/screenshots/home_page.png)<br>*Hero banner, key diagnostic metrics, and pipeline overview.* |
| **2. Clinical Intake Form (`/predict`)** | ![Intake Form](documentation/screenshots/predict_form.png)<br>*13 clinical biomarkers with 1-click Demo Profile Fillers.* |
| **3. Risk Assessment Report (`/predict` POST)** | ![Diagnostic Result](documentation/screenshots/result_page.png)<br>*Color-coded risk level, probability meter, and medical recommendations.* |
| **4. Healthcare Analytics Dashboard (`/dashboard`)** | ![Dashboard](documentation/screenshots/dashboard_page.png)<br>*Embedded ROC curves, PR curves, confusion matrices, and benchmark tables.* |

---

## 12. Future Enhancements

* **Deep Learning Integration**: Incorporate an Artificial Neural Network (ANN) and 1D Convolutional Neural Network (CNN) for raw continuous ECG waveform signal analysis.
* **SHAP / LIME Explainability**: Embed local SHAP waterfall plots on the prediction result page to explain feature impact per individual patient.
* **HL7 / FHIR Interoperability**: Support direct ingestion of electronic health records (EHR) via standard clinical data exchange protocols.
* **Cloud Deployment**: Containerize with Docker and deploy to AWS Elastic Beanstalk or Google Cloud Run behind an NGINX reverse proxy.

---

## 13. Conclusion

CardioPredict demonstrates the full lifecycle of a biomedical data science project—from raw data ingestion, median-based data cleansing, and inferential statistics to machine learning benchmarking and an interactive Flask deployment. By prioritizing clinical sensitivity without sacrificing specificity, the platform provides a reliable, interpretable tool for coronary artery disease risk assessment.

---

## 14. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed with scientific rigor for the **Introduction to Data Science (IDS)** Academic Review.
