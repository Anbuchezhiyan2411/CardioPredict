# CardioPredict: Exhaustive Project Structure & Codebase Guide

This document details every directory, file, script, and artifact present in the **CardioPredict** repository.

---

## 1. Directory Tree Overview

```
CardioPredict/
│
├── app.py                         # Primary Flask application controller & router
├── requirements.txt               # Pinned package dependencies
├── README.md                      # GitHub repository front-page overview
│
├── dataset/                       # Clinical raw and cleansed datasets
│   ├── processed.cleveland.data   # Raw UCI Cleveland dataset
│   ├── heart_cleaned.csv          # Cleaned dataset with imputed values & binary target
│   ├── cleaned_cardiopredict.csv  # Cleansed dataset backup
│   └── heart-disease.names        # Original UCI documentation & attribute notes
│
├── pipeline/                      # Modular Python machine learning pipeline
│   ├── __init__.py                # Package initialization
│   ├── utils.py                   # Path resolution and serialization helpers
│   ├── data_loader.py             # Data ingestion and (X, y) separation
│   ├── preprocessing.py           # Stratified train/test splitting & StandardScaler
│   ├── feature_engineering.py     # Variable taxonomy and feature selection rules
│   ├── train.py                   # Multi-algorithm model training orchestration
│   ├── evaluate.py                # Model benchmarking, metrics calculation, and plotting
│   └── predict.py                 # Production inference engine for web application
│
├── models/                        # Serialised trained estimators and preprocessors
│   ├── best_model.pkl             # Champion Random Forest model (90.16% accuracy)
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   ├── encoder.pkl                # Categorical mapping dictionaries
│   ├── feature_columns.pkl        # List of 13 input features in exact model order
│   ├── logistic_regression.pkl    # Serialised Logistic Regression model
│   ├── decision_tree.pkl          # Serialised Decision Tree model
│   ├── random_forest.pkl          # Serialised Random Forest model
│   ├── svm.pkl                    # Serialised Support Vector Machine model
│   └── knn.pkl                    # Serialised K-Nearest Neighbors model
│
├── evaluation/                    # Benchmarking artifacts and numerical exports
│   ├── model_comparison.csv       # Comparative metrics table for all 5 algorithms
│   ├── classification_reports.txt # Raw precision, recall, and f1 text logs
│   └── evaluation_summary.txt     # Executive performance summary
│
├── templates/                     # Flask Jinja2 HTML templates
│   ├── base.html                  # Master layout template (navbar, alerts, footer)
│   ├── home.html                  # Landing page (hero banner, KPI badges, pipeline cards)
│   ├── predict.html               # 13-feature clinical patient intake form
│   ├── result.html                # Diagnostic risk report & lifestyle recommendations
│   ├── dashboard.html             # Analytics dashboard with embedded visual benchmarks
│   ├── about.html                 # Dataset origins, methodology, and tech stack
│   └── error.html                 # Custom 404 / 500 error display
│
├── static/                        # Web application static assets
│   ├── css/
│   │   └── style.css              # Custom healthcare CSS theme & animations
│   ├── js/
│   │   └── script.js              # Client-side form helpers, tooltips, demo profile fillers
│   └── images/                    # Visualizations generated during Phases 4 & 8
│       ├── 01_histogram_age.png   # Age distribution plot
│       ├── 02_countplot_gender.png # Gender distribution plot
│       ├── 03_barchart_chestpain.png # Chest pain breakdown plot
│       ├── 04_piechart_target.png # Binary target distribution plot
│       ├── 05_boxplot_clinical_features.png # Clinical biomarker boxplots
│       ├── 06_violinplot_maxhr.png # Max heart rate violin plot
│       ├── 07_scatterplot_age_maxhr.png # Age vs Max heart rate scatter plot
│       ├── 08_pairplot_multivariate.png # Multivariate feature pairplot
│       ├── 09_correlation_heatmap.png # Pearson correlation heatmap
│       ├── 10_distribution_oldpeak.png # ST depression distribution plot
│       ├── confusion_matrix_*.png # 5 individual confusion matrix heatmaps
│       ├── roc_curve.png          # Multi-model ROC comparison curve
│       ├── precision_recall_curve.png # Multi-model Precision-Recall curve
│       ├── model_accuracy_comparison.png # Accuracy & F1 bar chart
│       └── feature_importance.png # Random Forest Gini feature importances
│
├── documentation/                 # Comprehensive academic and project documentation
│   ├── Project_Report.md          # Comprehensive academic research and project report
│   ├── Viva_Questions.md          # 50 technical viva defense questions and answers
│   ├── Requirements.md            # Hardware, software, and library requirements
│   ├── User_Guide.md              # Installation, operational, and troubleshooting manual
│   ├── Project_Structure.md       # (This file) Complete codebase directory guide
│   └── Final_Summary.md           # End-to-end summary of all 10 project phases
│
└── notebooks/                     # Interactive Jupyter Notebooks
    └── CardioPredict_EDA.ipynb    # Comprehensive notebook covering Phases 1 to 8
```

---

## 2. Detailed File Descriptions

### Root Directory
* **`app.py`**: The entry point for the Flask web application. Initializes the Flask server, registers HTTP route handlers (`/`, `/predict`, `/dashboard`, `/about`), manages user session flash messages, handles form submissions, and dispatches data to the inference engine.
* **`requirements.txt`**: Specifies all required Python packages with version pinning (e.g., `Flask>=3.0.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`).
* **`README.md`**: Professional GitHub repository landing document, containing badges, an executive overview, installation guides, benchmarking tables, and architectural diagrams.

---

### `dataset/` Directory
* **`processed.cleveland.data`**: The original comma-delimited clinical dataset from the UCI Machine Learning Repository (303 rows, 14 unlabelled columns, containing `'?'` sentinel values).
* **`heart_cleaned.csv`**: The cleansed dataset produced in Phase 2. Features imputed missing values (`CA` and `Thal` medians), binary target mapping ($0 = \text{Healthy}, 1 = \text{Disease}$), and clean column headers.
* **`heart-disease.names`**: Original dataset documentation by Dr. Robert Detrano, detailing contributor institutions, clinical attribute definitions, and experimental contexts.

---

### `pipeline/` Directory (Modular ML Architecture)
* **`utils.py`**: Central utility module providing absolute path resolution (`get_project_root()`, `get_models_dir()`) to prevent broken relative imports across execution environments, along with robust `joblib` save and load wrappers.
* **`data_loader.py`**: Reads `dataset/heart_cleaned.csv`, validates shape and column integrity, and partitions data into feature matrix $X$ and target vector $y$.
* **`preprocessing.py`**: Executes an 80/20 stratified split (`stratify=y`, random state 42) into $X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$. Fits `StandardScaler` strictly on $X_{\text{train}}$, transforms $X_{\text{test}}$, and serializes `models/scaler.pkl` and `models/feature_columns.pkl`.
* **`feature_engineering.py`**: Documents attribute taxonomies (numerical, binary, nominal, ordinal), evaluates feature interaction necessity, and validates retaining all 13 clinical biomarkers.
* **`train.py`**: Trains five supervised learning algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, KNN), records train/test accuracy and runtime, and serializes each estimator into `models/`.
* **`evaluate.py`**: Benchmarking module. Evaluates all 5 models across 8 diagnostic metrics, generates classification reports, confusion matrix heatmaps, ROC curves, PR curves, and exports `evaluation/model_comparison.csv`.
* **`predict.py`**: Production inference engine. Implements the `CardioPredictor` singleton class that loads `best_model.pkl` and `scaler.pkl` to run instant, zero-retraining inference for incoming web requests.

---

### `models/` Directory (Serialised Artifacts)
* **`best_model.pkl`**: Serialised champion estimator (**Random Forest Classifier**, 100 trees, 90.16% test accuracy, 92.86% recall).
* **`scaler.pkl`**: Fitted `StandardScaler` containing training feature means ($\mu$) and standard deviations ($\sigma$).
* **`feature_columns.pkl`**: Serialised list of 13 feature names ensuring strict column order during inference.
* **`encoder.pkl`**: Encoded feature metadata dictionary.
* **`logistic_regression.pkl`**, **`decision_tree.pkl`**, **`random_forest.pkl`**, **`svm.pkl`**, **`knn.pkl`**: Standalone serialised models saved during training.

---

### `templates/` Directory (Jinja2 Web Views)
* **`base.html`**: Master template providing the responsive HTML5 skeleton, Bootstrap 5 navbar, flash notification container, and healthcare footer.
* **`home.html`**: Landing page showcasing the hero section, key clinical metrics, interactive call-to-action buttons, and the 4-stage pipeline overview.
* **`predict.html`**: Clinical intake form for all 13 patient biomarkers, equipped with 1-click **Demo Patient Fillers** for testing.
* **`result.html`**: Diagnostic risk report card displaying risk tier (Low, Moderate, High), animated probability gauge, tailored medical recommendations, and submitted parameter logs.
* **`dashboard.html`**: Analytics portal featuring executive KPI cards, an interactive model comparison table, and embedded EDA and evaluation plots.
* **`about.html`**: Detailed background on the Cleveland Clinic dataset, machine learning methodologies, and technical stack.
* **`error.html`**: Custom 404 / 500 error page handling broken URLs or server exceptions gracefully.

---

### `static/` Directory (Styles, Scripts & Visualizations)
* **`css/style.css`**: Custom CSS stylesheet defining the medical color palette (primary `#0d6efd`, pulse `#e63946`), card drop shadows, smooth transitions, and responsive tables.
* **`js/script.js`**: Client-side JavaScript implementing Bootstrap tooltips, form validation, and 1-click demo case loaders.
* **`images/`**: Contains the 10 EDA plots generated in Phase 4 and the 9 machine learning evaluation graphics generated in Phase 8.

---

### `documentation/` Directory
* **`Project_Report.md`**: Comprehensive academic project report formatted for college capstone submission.
* **`Viva_Questions.md`**: 50 technical viva defense questions with comprehensive answers.
* **`Requirements.md`**: Hardware, software, environment, and library specifications.
* **`User_Guide.md`**: Installation, execution, prediction, and troubleshooting instructions.
* **`Project_Structure.md`**: (This document) Exhaustive codebase directory guide.
* **`Final_Summary.md`**: High-level summary of all 10 project phases.

---

### `notebooks/` Directory
* **`CardioPredict_EDA.ipynb`**: Complete, self-contained Jupyter Notebook providing interactive code, output tables, and visualizations spanning Phases 1 through 8.
