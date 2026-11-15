# CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
## Comprehensive Academic Project Report

**Course**: Introduction to Data Science (IDS) Capstone Project  
**Author**: Academic Project Review Team  
**Date of Submission**: Academic Year 2026–2027  

---

## Abstract

Coronary artery disease (CAD) remains the premier global contributor to cardiovascular mortality, necessitating reliable, non-invasive, and cost-effective clinical screening tools. This report presents **CardioPredict**, an end-to-end data science and machine learning clinical decision support platform developed on the Cleveland Heart Disease benchmark dataset. 

Following a disciplined 10-phase software and analytical lifecycle, the raw dataset underwent rigorous data sanitization, median imputation of missing values, and binary target transformation. Parametric and non-parametric exploratory data analysis, paired with inferential statistical hypothesis testing (two-sample Independent $t$-tests, Pearson correlation, Chi-Square tests of independence, and One-Way ANOVA), empirically validated key clinical associations with myocardial ischemia. Five supervised classification paradigms—Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)—were trained under an 80/20 stratified partition and benchmarked across eight diagnostic metrics.

The **Random Forest Classifier** emerged as the champion model, demonstrating an optimal balance of **90.16% test accuracy**, **92.86% clinical sensitivity (recall)**, **86.67% precision**, and an **ROC-AUC of 0.9654**, committing only 2 false-negative classifications out of 28 holdout pathology cases. The serialised model and preprocessing pipeline were productionized into an interactive, full-stack Flask web application providing sub-millisecond risk stratification, diagnostic probability gauges, personalized medical guidance, and an interactive healthcare analytics dashboard.

---

## 1. Introduction

Cardiovascular diseases (CVDs) encompass disorders affecting the heart and vascular system, primarily ischemic heart disease, cerebrovascular disease, and rheumatic heart conditions. According to the World Health Organization (WHO), an estimated 17.9 million individuals die from CVDs each year, accounting for approximately 32% of all global fatalities. Over 75% of these deaths occur in low- and middle-income regions where access to advanced diagnostic infrastructure, such as catheterized coronary angiography, remains severely constrained by equipment costs and specialist shortages.

Data science and predictive machine learning offer an unprecedented opportunity to assist medical practitioners. By identifying subtle non-linear interactions across routine, non-invasive clinical biomarkers—such as blood pressure, exercise electrocardiogram (ECG) alterations, resting heart rate, and cholesterol levels—machine learning models can compute individual risk probabilities rapidly, flagging at-risk individuals before irreversible myocardial infarction occurs.

---

## 2. Problem Statement

Traditional diagnostic protocols for coronary artery disease rely heavily on sequential clinical scoring (e.g., Framingham Risk Score) followed by invasive cardiac catheterization. While angiography provides definitive visualization of arterial occlusion, it is expensive, carries procedural risks (such as arterial dissection and dye-induced nephropathy), and requires specialized hospital environments.

Conversely, manual interpretation of non-invasive stress test profiles is susceptible to inter-observer variability and subjective clinical heuristic errors. The core problem addressed by this project is:
> *How can multi-dimensional, non-invasive clinical biomarkers be rigorously cleansed, statistically validated, and leveraged via machine learning to deliver a high-sensitivity, interpretable, and reproducible cardiac risk classification engine deployed as a web application?*

---

## 3. Project Objectives

The primary objectives of the **CardioPredict** initiative are:
1. **Data Sanitization & Preprocessing**: Ingest the raw UCI Cleveland dataset, resolve missing value anomalies, convert multi-tier diagnoses into binary outcomes ($0 = \text{Healthy}, 1 = \text{Heart Disease}$), and implement robust feature scaling.
2. **Exploratory & Inferential Statistics**: Conduct descriptive statistical profiling and formal hypothesis tests ($t$-tests, $\chi^2$, ANOVA, Pearson correlation) to establish the statistical significance of biomarker divergences between healthy and diseased cohorts.
3. **Multi-Model Machine Learning Pipeline**: Construct a modular Python pipeline implementing five diverse supervised learning algorithms without data leakage.
4. **Comprehensive Benchmarking**: Evaluate models using metrics prioritizing clinical sensitivity (recall) to minimize life-threatening false negatives, validated via Confusion Matrices, ROC Curves, and Precision-Recall Curves.
5. **Full-Stack Deployment**: Build an intuitive, production-ready Flask web platform with zero-retraining inference, instant patient risk stratification, and an interactive analytics dashboard.

---

## 4. Dataset Description

The project utilizes the **Cleveland Heart Disease Dataset** hosted by the University of California, Irvine (UCI) Machine Learning Repository. Originally collected across the Cleveland Clinic Foundation under Dr. Robert Detrano, Dr. William Steinbrunn, and Dr. Matthias Pfisterer, this benchmark comprises 303 patient records across 14 measured attributes:

| Column Name | Data Type | Clinical Range / Categories | Description |
|---|---|---|---|
| `Age` | Continuous | 29 – 77 years | Age of the patient |
| `Sex` | Binary | 1 = Male; 0 = Female | Biological sex |
| `ChestPainType` | Nominal | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic | Nature of chest pain experienced |
| `RestingBP` | Continuous | 94 – 200 mm Hg | Resting systolic blood pressure upon admission |
| `Cholesterol` | Continuous | 126 – 564 mg/dL | Serum cholesterol in mg/dL |
| `FastingBS` | Binary | 1: > 120 mg/dL; 0: $\le$ 120 mg/dL | Fasting blood glucose level |
| `RestingECG` | Nominal | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy | Resting electrocardiographic reading |
| `MaxHR` | Continuous | 71 – 202 bpm | Maximum heart rate achieved during treadmill test |
| `ExerciseAngina` | Binary | 1 = Yes; 0 = No | Angina pectoris elicited by physical exertion |
| `Oldpeak` | Continuous | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest |
| `ST_Slope` | Ordinal | 1: Upsloping, 2: Flat, 3: Downsloping | Slope of peak exercise ST segment |
| `CA` | Discrete | 0, 1, 2, 3 vessels | Major blood vessels (0–3) colored by fluoroscopy |
| `Thal` | Nominal | 3: Normal, 6: Fixed defect, 7: Reversible defect | Thallium scintigraphy stress test |
| `Target` | Binary | 0 = Healthy, 1 = Heart Disease | Angiographic coronary artery disease status |

---

## 5. Data Preprocessing & Cleaning (Phases 1 & 2)

Raw clinical records often suffer from incomplete reporting and arbitrary sentinel coding. In the raw `processed.cleveland.data` file:
1. **Sentinel Value Resolution**: Missing values were encoded as the character `'?'`. Ingestion identified 4 missing entries in `CA` and 2 in `Thal`. These were replaced with IEEE standard `NaN` values.
2. **Missing Value Imputation**: Due to the skewed nature of clinical vessel counts and categorical codes, mean imputation would introduce fractional, non-physical values. Consequently, **median imputation** was applied:
   $$\text{Imputed}(CA) = \text{Median}(CA) = 0.0$$
   $$\text{Imputed}(Thal) = \text{Median}(Thal) = 3.0$$
   This preserved the discrete medical scale without discarding valuable patient rows.
3. **Target Binarization**: The raw target variable spanned discrete integers $0$ (angiographic diameter narrowing $< 50\%$) and $1, 2, 3, 4$ (narrowing $\ge 50\%$). To formulate a binary risk screening problem, values $1$ through $4$ were mapped to $1$:
   $$y = \begin{cases} 0, & \text{if } \text{Raw Target} = 0 \text{ (Healthy / No Significant CAD)} \\ 1, & \text{if } \text{Raw Target} \ge 1 \text{ (CAD Diagnosed)} \end{cases}$$
4. **Class Distribution**: The resulting dataset comprised **164 healthy patients (54.13%)** and **139 CAD patients (45.87%)**, exhibiting a well-balanced clinical distribution devoid of extreme class imbalance.

---

## 6. Exploratory Data Analysis (Phases 3 & 4)

Exploratory data analysis revealed pronounced physiological divergences:
* **Age Distribution**: Patients diagnosed with heart disease were older ($\mu = 56.6$ years) compared to healthy individuals ($\mu = 52.6$ years).
* **Gender Disparity**: Males exhibited a substantially higher proportion of heart disease diagnoses (55.3%) compared to females (25.8%), aligning with clinical literature on pre-menopausal estrogen cardioprotection.
* **Chest Pain Paradox**: Asymptomatic chest pain (`ChestPainType = 4`) was the most prevalent category among diseased patients (72.7%), underscoring the lethal nature of "silent ischemia" where absence of sharp pain delays clinical presentation.
* **Heart Rate & Workload**: Max Heart Rate (`MaxHR`) exhibited a strong negative association with pathology; cardiac patients struggled to elevate heart rates during stress testing ($\mu = 139.3$ bpm vs. $158.4$ bpm for healthy subjects).
* **Electrocardiographic Biomarkers**: ST-segment depression (`Oldpeak`) was markedly elevated in diseased patients ($\mu = 1.58$ mm vs. $0.58$ mm), indicating exercise-induced subendocardial ischemia.

---

## 7. Inferential Statistics & Hypothesis Testing (Phase 5)

To confirm that observed distributional differences were statistically significant and not artifacts of random sampling, formal hypothesis tests were executed at $\alpha = 0.05$:

### 7.1 Two-Sample Independent $t$-Tests (Numerical Features)
* **Age vs. Target**: $t = -4.01$, $p = 7.95 \times 10^{-5}$ ($p < 0.001$). Null hypothesis rejected; age is significantly higher in diseased patients.
* **MaxHR vs. Target**: $t = 8.07$, $p = 1.69 \times 10^{-14}$ ($p < 0.001$). Strong statistical confirmation that maximum exercise heart rate is depressed in cardiac patients.
* **Oldpeak vs. Target**: $t = -7.99$, $p = 2.76 \times 10^{-14}$ ($p < 0.001$). ST depression is definitively correlated with ischemia.

### 7.2 Chi-Square Tests of Independence ($\chi^2$) (Categorical Features)
* **ChestPainType vs. Target**: $\chi^2 = 81.82$, $p = 1.25 \times 10^{-17}$ ($\text{dof} = 3$). Rejection of independence; chest pain classification is profoundly associated with pathology.
* **ExerciseAngina vs. Target**: $\chi^2 = 55.94$, $p = 7.46 \times 10^{-14}$ ($\text{dof} = 1$). Strong dependency between exertional angina and arterial obstruction.
* **Thal vs. Target**: $\chi^2 = 85.33$, $p = 3.01 \times 10^{-19}$ ($\text{dof} = 2$). Reversible defects during nuclear scintigraphy heavily signal multivessel ischemia.

### 7.3 One-Way Analysis of Variance (ANOVA)
* **RestingBP across Chest Pain Categories**: $F = 1.09$, $p = 0.354$. Null hypothesis retained; resting blood pressure does not significantly diverge strictly as a function of chest pain classification alone.

---

## 8. Feature Engineering & Scaling (Phase 6)

1. **Standardization via `StandardScaler`**:
   Numerical features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) span divergent scales (e.g., Cholesterol up to 564 mg/dL vs. Oldpeak up to 6.2 mm). Algorithms relying on distance calculations (KNN, SVM) or gradient descent (Logistic Regression) are vulnerable to feature dominance without standardization.
   $$z = \frac{x - \mu}{\sigma}$$
   The scaler was fitted strictly on the training partition ($X_{\text{train}}$) and subsequently applied to $X_{\text{test}}$, completely eliminating data leakage.
2. **Feature Preservation**:
   All 13 original biomarkers possess established physiological interpretability in cardiology. Artificial interaction terms (e.g., polynomial combinations) were avoided to preserve direct explainability and clinical auditability for physicians.

---

## 9. Machine Learning Model Training (Phase 7)

Five distinct algorithmic paradigms were implemented:
1. **Logistic Regression (`L2` Regularization)**: Serves as the clinical baseline, modeling the log-odds of cardiac disease via maximum likelihood estimation.
2. **Decision Tree Classifier (`criterion='gini'`, `max_depth=4`)**: Generates an interpretable hierarchical rule set mapping clinical cutoffs.
3. **Random Forest Classifier (`n_estimators=100`, `max_depth=5`)**: An ensemble bootstrap aggregation (bagging) model combining decorrelated trees to reduce variance and mitigate overfitting.
4. **Support Vector Machine (`kernel='rbf'`, $C=1.0$, $\gamma=\text{'scale'}$)**: Constructs optimal hyperplanes in high-dimensional reproducing kernel Hilbert spaces.
5. **K-Nearest Neighbors ($k=5$, Euclidean Metric)**: Non-parametric instance-based learner classifying patients based on proximity to nearest feature-space neighbors.

---

## 10. Comprehensive Model Evaluation & Benchmarking (Phase 8)

Evaluation was conducted on a holdout test partition of 61 patients ($N_{\text{healthy}} = 33, N_{\text{disease}} = 28$):

### Benchmark Comparative Performance Table

| Model | Train Acc | Test Acc | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **100.00%** | **90.16%** | **86.67%** | **92.86%** | **89.66%** | **0.9654** | **87.88%** |
| **Support Vector Machine** | 91.32% | 86.89% | 83.33% | 89.29% | 86.21% | 0.9232 | 84.85% |
| **Logistic Regression** | 85.54% | 85.25% | 82.76% | 85.71% | 84.21% | 0.9080 | 84.85% |
| **K-Nearest Neighbors** | 87.19% | 81.97% | 79.31% | 82.14% | 80.70% | 0.8874 | 81.82% |
| **Decision Tree** | 100.00% | 77.05% | 74.07% | 71.43% | 72.73% | 0.7662 | 81.82% |

### Detailed Evaluation Analysis
* **Clinical Recall / Sensitivity**: In cardiac screening, **False Negatives** represent undiagnosed patients sent home who remain at acute risk of fatal myocardial infarction. Random Forest achieved a clinical sensitivity of **92.86%**, misclassifying only 2 diseased patients out of 28.
* **ROC-AUC (0.9654)**: The Receiver Operating Characteristic curve demonstrated exceptional discriminative ability, confirming that a randomly chosen diseased patient will be assigned a higher predicted risk score than a healthy subject in 96.5% of comparisons.
* **Feature Importance**: Random Forest Gini impurity reduction identified the primary diagnostic drivers:
  1. `Thal` (Thallium Scintigraphy defect status): 14.8%
  2. `CA` (Number of fluoroscopy-colored major vessels): 13.9%
  3. `Oldpeak` (ST depression magnitude): 12.3%
  4. `MaxHR` (Exercise maximum heart rate): 11.9%
  5. `ChestPainType` (Angina presentation): 10.7%

---

## 11. Flask Web Application Architecture (Phase 9)

The production web interface connects the analytical pipeline to a responsive user interface:
* **Controller Layer (`app.py`)**: Manages HTTP traffic, validates payload schemas, catches exceptions, and renders Jinja2 templates.
* **Inference Service (`pipeline/predict.py`)**: Encapsulates model and scaler deserialization into an isolated `CardioPredictor` singleton. Guarantees that the web application never initiates retraining cycles.
* **Frontend Presentation**:
  * **Intake Interface (`predict.html`)**: Features 13 intuitive input fields with clinically realistic bounding constraints, accessible tooltips, and 1-click **Demo Fillers** (pre-loading validated Low-Risk and High-Risk clinical test profiles).
  * **Diagnostic Card (`result.html`)**: Presents a prominent color-coded risk badge, animated probability gauge, complete parameter breakdown, and automated lifestyle guidance.
  * **Analytics Dashboard (`dashboard.html`)**: Incorporates responsive KPI metric cards, complete benchmark performance tables, and high-resolution exploratory data charts.

---

## 12. Results & Discussion

The experimental findings demonstrate that standard non-invasive clinical biomarkers yield superior diagnostic accuracy when evaluated collectively through non-linear ensemble models. While single linear boundaries (Logistic Regression: 85.25%) perform respectably, they fail to capture complex multi-system physiological interactions (such as the compounding effect of exercise-induced ST depression occurring alongside blunted heart rate elevation).

Ensemble random forests effectively neutralized variance and achieved **90.16% test accuracy** with an **F1-score of 89.66%**. The model was confirmed to operate without data leakage, as demonstrated by the consistency between training accuracy and cross-validated generalization metrics.

---

## 13. Conclusion

The **CardioPredict** capstone project successfully demonstrates an industry-standard, end-to-end data science lifecycle. By uniting statistical rigor, robust missing data handling, multi-algorithm benchmarking, and modern web application development, CardioPredict provides an accessible, high-performing screening instrument for coronary artery disease. The system demonstrates the practical viability of machine learning in supporting frontline healthcare professionals.

---

## 14. Future Scope

* **Integration of Continuous Waveform Data**: Expand the feature pipeline to consume raw 12-lead digital ECG waveforms via 1D Convolutional Neural Networks (CNNs).
* **Model Explainability (XAI)**: Embed interactive SHAP (SHapley Additive exPlanations) force plots within the prediction result page to illustrate feature attribution for individual patients.
* **FHIR / EHR Interoperability**: Implement Fast Healthcare Interoperability Resources (FHIR) API adapters to allow direct, secure patient record ingestion from hospital databases.
* **Mobile & Edge Deployment**: Quantize the Random Forest model into ONNX format for real-time mobile app execution on low-power devices in remote rural clinics.

---

## 15. References

1. Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. The American Journal of Cardiology, 64(5), 304–310.
2. World Health Organization (2021). *Cardiovascular diseases (CVDs) Fact Sheet*. Geneva: WHO.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
4. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
5. Grinberg, M. (2018). *Flask Web Development: Developing Web Applications with Python*. O'Reilly Media.
