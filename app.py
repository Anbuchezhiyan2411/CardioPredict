"""CardioPredict: Flask Web Application.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : app.py
Purpose : Production WSGI entry point serving the clinical prediction engine,
          analytics dashboard, exploratory visual galleries, and about pages.
=============================================================================
"""

import os
import pandas as pd
from typing import Any, Dict, List
from flask import Flask, render_template, request, redirect, url_for, flash

try:
    from pipeline.predict import make_prediction
    from pipeline.utils import get_project_root
except ImportError:
    from pipeline.predict import make_prediction
    from pipeline.utils import get_project_root

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "cardiopredict-secure-clinical-key-2026")

# Resolve absolute paths to project directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")


def get_dashboard_data() -> Dict[str, Any]:
    """Load model benchmarking metrics and textual summaries for the dashboard.

    Returns:
        Dict[str, Any]: Parsed comparison table rows and executive summary text.
    """
    csv_path = os.path.join(EVALUATION_DIR, "model_comparison.csv")
    summary_path = os.path.join(EVALUATION_DIR, "evaluation_summary.txt")

    comparison_records: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        comparison_df = pd.read_csv(csv_path)
        comparison_records = comparison_df.to_dict(orient="records")

    summary_content = "Evaluation summary not yet compiled. Run Phase 8 evaluation script."
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_content = f.read()

    return {
        "model_comparison": comparison_records,
        "evaluation_summary": summary_content
    }


# =============================================================================
# WEB APPLICATION ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    """Render the landing home page with hero section, KPIs, and workflow preview."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle patient intake form display (GET) and inference submission (POST)."""
    if request.method == "POST":
        try:
            # Extract form payload
            raw_form_data = {
                "Age": request.form.get("Age"),
                "Sex": request.form.get("Sex"),
                "ChestPainType": request.form.get("ChestPainType"),
                "RestingBP": request.form.get("RestingBP"),
                "Cholesterol": request.form.get("Cholesterol"),
                "FastingBS": request.form.get("FastingBS"),
                "RestingECG": request.form.get("RestingECG"),
                "MaxHR": request.form.get("MaxHR"),
                "ExerciseAngina": request.form.get("ExerciseAngina"),
                "Oldpeak": request.form.get("Oldpeak"),
                "ST_Slope": request.form.get("ST_Slope"),
                "CA": request.form.get("CA"),
                "Thal": request.form.get("Thal")
            }

            # Execute real-time prediction using pipeline/predict.py
            diagnostic_result = make_prediction(raw_form_data)

            # Render clinical result report
            return render_template("result.html", result=diagnostic_result)

        except Exception as error:
            flash(f"Clinical Processing Error: {str(error)}", "danger")
            return redirect(url_for("predict"))

    # GET request: Display blank clinical intake form
    return render_template("predict.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Render the cardiovascular analytics dashboard with EDA and ML benchmarks."""
    dashboard_data = get_dashboard_data()
    return render_template(
        "dashboard.html",
        model_comparison=dashboard_data["model_comparison"],
        evaluation_summary=dashboard_data["evaluation_summary"]
    )


@app.route("/about", methods=["GET"])
def about():
    """Render the project architecture, dataset provenance, and technology stack."""
    return render_template("about.html")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def page_not_found(e):
    """Render custom 404 page for missing endpoints."""
    return render_template(
        "error.html",
        error_code="404",
        error_title="Clinical Endpoint Not Found",
        error_message="The requested page, patient record, or clinical resource does not exist."
    ), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Render custom 500 page for internal server faults."""
    return render_template(
        "error.html",
        error_code="500",
        error_title="Internal Processing Error",
        error_message="A server-side execution fault occurred. Please verify input data and retry."
    ), 500


if __name__ == "__main__":
    print("=" * 70)
    print("   🫀 CardioPredict Clinical Web Server Starting...")
    print("   Access Dashboard & Prediction UI: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
