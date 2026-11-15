# CardioPredict: User & Operational Guide

Welcome to the **CardioPredict** operational manual. This step-by-step user guide provides instructions on how to install, launch, operate, and troubleshoot the web application.

---

## 1. Installation Guide

### Prerequisites
Ensure your workstation has **Python 3.10 or higher** installed.

### Step 1: Open Terminal / Shell
* **Windows**: Open PowerShell or Command Prompt.
* **macOS / Linux**: Open Terminal.

Navigate to your desired project directory:
```bash
cd c:\Anbu\CardioPredict
```

### Step 2: Create a Virtual Environment (Recommended)
Isolating dependencies ensures no conflicts with system-wide Python libraries:

```powershell
# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

*Note: If PowerShell displays an `Execution_Policies` script error, run:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3: Install Required Dependencies
Install all required libraries using the provided manifest:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Launching the Web Application

With your virtual environment active, run the Flask application:

```bash
python app.py
```

### Expected Console Output:
```
✅ CardioPredictor: Inference engine initialized successfully.
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Accessing the Web Interface
Open any modern web browser (Chrome, Firefox, Edge, Safari) and navigate to:
```
http://127.0.0.1:5000
```

---

## 3. Navigating the Application

The web portal features a top navigation bar providing access to four primary modules:
1. **Home (`/`)**: Project motivation, diagnostic highlights, and multi-stage pipeline cards.
2. **Predict (`/predict`)**: 13-feature clinical patient intake portal.
3. **Analytics Dashboard (`/dashboard`)**: Model benchmarks, performance tables, and charts.
4. **About (`/about`)**: Dataset documentation, team information, and architectural overview.

---

## 4. How to Make a Patient Prediction

1. Click **"Start Prediction"** on the home page or navigate to the **Predict** tab in the top navigation bar.
2. You will be presented with the **13 Clinical Biomarker Intake Form**:
   * **Age**: Patient's age in years (range: 20 to 100).
   * **Biological Sex**: Select Male or Female.
   * **Chest Pain Type**: Select Typical Angina, Atypical Angina, Non-Anginal, or Asymptomatic.
   * **Resting Blood Pressure**: Systolic blood pressure in mm Hg (range: 80 to 220).
   * **Serum Cholesterol**: Total cholesterol in mg/dL (range: 100 to 600).
   * **Fasting Blood Sugar**: Select whether fasting blood sugar exceeds 120 mg/dL.
   * **Resting ECG**: Normal, ST-T Wave Abnormality, or Left Ventricular Hypertrophy.
   * **Maximum Heart Rate**: Peak heart rate achieved during exercise (range: 60 to 220 bpm).
   * **Exercise Induced Angina**: Select Yes or No.
   * **ST Depression (`Oldpeak`)**: Exercise-induced ST depression in mm (range: 0.0 to 6.5).
   * **ST Slope**: Upsloping, Flat, or Downsloping.
   * **Major Vessels (`CA`)**: Number of colored major coronary vessels (0, 1, 2, or 3).
   * **Thalassemia Scintigraphy (`Thal`)**: Normal, Fixed Defect, or Reversible Defect.

### Instant Testing with 1-Click Demo Fillers
For demonstration purposes, click either button located above the form:
* **"Demo: Low Risk Patient"**: Populates the form with a verified healthy 45-year-old patient profile.
* **"Demo: High Risk Patient"**: Populates the form with a verified severe CAD 67-year-old patient profile.

3. Click the blue **"Analyze Cardiac Risk"** button.

### Interpreting the Diagnostic Result
The application instantly processes the inputs through `pipeline/predict.py` and renders `result.html`:
* **Risk Badge**:
  * **Low Risk / Healthy (Green)**: Probability $< 50\%$.
  * **Moderate Risk (Yellow)**: Probability $50\% \le p < 75\%$.
  * **High Risk / Critical (Red)**: Probability $\ge 75\%$.
* **Probability Gauge**: Visual progress bar depicting the exact probability calculated by the Random Forest model.
* **Key Clinical Insights & Next Steps**: Automated, patient-specific advice (e.g., diet modifications for elevated cholesterol, stress echo testing for high ST depression).
* **Summary Table**: Full audit log of all 13 entered values for medical records.

---

## 5. How to Use the Analytics Dashboard

Navigate to the **Dashboard** link in the navigation bar to review clinical benchmarking:
1. **Executive KPI Cards**: Review high-level metrics (Champion Accuracy: 90.16%, Recall: 92.86%, Precision: 86.67%, ROC-AUC: 0.9654).
2. **Model Comparison Table**: Compare all 5 evaluated algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, KNN) across 8 metrics.
3. **Exploratory Data Analysis Gallery**: Review distributions of Age, Sex, Cholesterol, Chest Pain classifications, and Correlation Heatmaps.
4. **Diagnostic Evaluation Curves**: Inspect ROC Curves, Precision-Recall Curves, Feature Importance rankings, and individual Confusion Matrices.

---

## 6. Troubleshooting Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'flask'` (or other package)
* **Cause**: Flask was installed into a different Python environment or the virtual environment is not active.
* **Resolution**:
  ```powershell
  # Ensure your virtual environment is active:
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### Issue 2: `Port 5000 is already in use`
* **Cause**: Another instance of Flask or an existing web application is bound to port 5000.
* **Resolution**:
  * **Option A**: Terminate the existing process via Task Manager (look for `python.exe`).
  * **Option B**: Change the port in `app.py` by launching with:
    ```bash
    python -c "from app import app; app.run(port=5001, debug=True)"
    ```
    Then visit `http://127.0.0.1:5001`.

### Issue 3: `FileNotFoundError: models/best_model.pkl not found`
* **Cause**: `app.py` was launched from a subfolder rather than the project root directory.
* **Resolution**: Make sure you navigate to the root directory `c:\Anbu\CardioPredict` before executing `python app.py`. `pipeline/utils.py` automatically resolves absolute paths relative to the project root.

### Issue 4: Form validation error: "Please enter a valid numeric value"
* **Cause**: A non-numeric character was entered into a numeric field (such as `Oldpeak` or `Cholesterol`).
* **Resolution**: Enter standard numbers (e.g., `210` for cholesterol, `1.5` for Oldpeak).

---

## 7. Stopping the Application

To shut down the local Flask server safely, return to your terminal window and press:
```
CTRL + C
```
The terminal prompt will reappear, and the server will cease listening on port 5000.
