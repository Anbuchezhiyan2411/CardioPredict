"""Prediction Service Module.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : pipeline/predict.py
Purpose : Inference engine for single-patient diagnostic prediction using the
          persisted Best Model (Random Forest) and StandardScaler from Phase 8.
=============================================================================
"""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd


class CardioPredictor:
    """Production inference engine for CardioPredict clinical diagnostics.

    Loads and caches the best model, feature scaler, and feature schema.
    Provides thread-safe prediction and risk stratification for incoming patients.
    """

    _instance: Optional["CardioPredictor"] = None

    def __init__(self, models_dir: Optional[str] = None):
        """Initialize the predictor by loading serialized artifacts.

        Args:
            models_dir (Optional[str]): Path to directory containing .pkl artifacts.
        """
        if models_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, "models")

        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.columns_path = os.path.join(models_dir, "feature_columns.pkl")
        self.encoder_path = os.path.join(models_dir, "encoder.pkl")

        # Load artifacts into memory
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load pickled artifacts from disk with error checking."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file missing: {self.model_path}. Run Phase 8 first.")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file missing: {self.scaler_path}. Run Phase 6 first.")
        if not os.path.exists(self.columns_path):
            raise FileNotFoundError(f"Feature columns schema missing: {self.columns_path}.")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_columns: List[str] = joblib.load(self.columns_path)
        self.encoder_metadata: Dict[str, Any] = (
            joblib.load(self.encoder_path) if os.path.exists(self.encoder_path) else {}
        )
        print("✅ CardioPredictor: Inference engine initialized successfully.")

    @classmethod
    def get_instance(cls) -> "CardioPredictor":
        """Singleton pattern accessor to prevent redundant disk I/O on each HTTP request."""
        if cls._instance is None:
            cls._instance = CardioPredictor()
        return cls._instance

    def predict_patient(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw clinical form inputs and return an evaluated diagnostic report.

        Args:
            form_data (Dict[str, Any]): Dictionary of raw inputs submitted via web form.

        Returns:
            Dict[str, Any]: Comprehensive diagnostic payload including:
                - prediction (int): 0 (Healthy) or 1 (Heart Disease)
                - prediction_label (str): Human-readable diagnosis
                - probability_disease (float): Probability of heart disease (0.0 to 1.0)
                - probability_healthy (float): Probability of healthy status (0.0 to 1.0)
                - confidence_pct (float): Confidence percentage
                - risk_level (str): 'Low', 'Moderate', or 'High'
                - risk_badge_class (str): Bootstrap alert/badge color class
                - clinical_recommendation (str): Practical medical advice
                - formatted_inputs (Dict[str, Any]): Cleaned input values for review
        """
        # 1. Type casting and sanitization of raw inputs
        try:
            cleaned_inputs = {
                "Age": float(form_data["Age"]),
                "Sex": int(form_data["Sex"]),
                "ChestPainType": int(form_data["ChestPainType"]),
                "RestingBP": float(form_data["RestingBP"]),
                "Cholesterol": float(form_data["Cholesterol"]),
                "FastingBS": int(form_data["FastingBS"]),
                "RestingECG": int(form_data["RestingECG"]),
                "MaxHR": float(form_data["MaxHR"]),
                "ExerciseAngina": int(form_data["ExerciseAngina"]),
                "Oldpeak": float(form_data["Oldpeak"]),
                "ST_Slope": int(form_data["ST_Slope"]),
                "CA": int(form_data["CA"]),
                "Thal": int(form_data["Thal"])
            }
        except (KeyError, ValueError) as err:
            raise ValueError(f"Invalid or missing patient input field: {err}")

        # 2. Build aligned single-row DataFrame matching the training schema
        input_df = pd.DataFrame([cleaned_inputs])[self.feature_columns]

        # 3. Standardize continuous numerical features using fitted StandardScaler
        input_scaled = self.scaler.transform(input_df)

        # 4. Predict binary class and posterior probabilities
        predicted_class = int(self.model.predict(input_scaled)[0])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_scaled)[0]
            prob_healthy = float(probabilities[0])
            prob_disease = float(probabilities[1])
        else:
            # Fallback for models without direct probability calibration
            prob_disease = 1.0 if predicted_class == 1 else 0.0
            prob_healthy = 1.0 - prob_disease

        # 5. Risk Stratification Logic
        if prob_disease < 0.35:
            risk_level = "Low Risk"
            risk_badge_class = "success"
            confidence_pct = prob_healthy * 100.0
            recommendation = (
                "Patient exhibits favorable cardiovascular markers with low probability "
                "of obstructive coronary stenosis. Recommend standard routine annual wellness check, "
                "balanced Mediterranean diet, and 150 minutes of weekly aerobic exercise."
            )
        elif 0.35 <= prob_disease < 0.65:
            risk_level = "Moderate / Borderline Risk"
            risk_badge_class = "warning"
            confidence_pct = max(prob_healthy, prob_disease) * 100.0
            recommendation = (
                "Borderline cardiovascular indicators observed. Several intermediate parameters "
                "(such as blood pressure, exercise response, or lipid profile) indicate emerging vulnerability. "
                "Recommend non-invasive 12-lead stress echocardiogram, fasting lipid fraction re-test, "
                "and lifestyle/dietary intervention within 3 months."
            )
        else:
            risk_level = "High Risk"
            risk_badge_class = "danger"
            confidence_pct = prob_disease * 100.0
            recommendation = (
                "URGENT: Patient demonstrates high-probability indicators of coronary artery disease "
                "(significant ST depression, reduced chronotropic capacity, or multiple compromised vessels). "
                "Immediate consultation with a board-certified cardiologist is advised. Recommend formal "
                "diagnostic coronary angiography and medical therapy evaluation."
            )

        diagnosis_label = "Heart Disease Present" if predicted_class == 1 else "Healthy (No Significant Disease)"

        # 6. Human-readable metadata mappings for the results card
        formatted_inputs = {
            "Age": f"{int(cleaned_inputs['Age'])} years",
            "Sex": "Male" if cleaned_inputs["Sex"] == 1 else "Female",
            "ChestPainType": {
                1: "Typical Angina (Type 1)",
                2: "Atypical Angina (Type 2)",
                3: "Non-Anginal Discomfort (Type 3)",
                4: "Asymptomatic (Type 4)"
            }.get(cleaned_inputs["ChestPainType"], str(cleaned_inputs["ChestPainType"])),
            "RestingBP": f"{int(cleaned_inputs['RestingBP'])} mm Hg",
            "Cholesterol": f"{int(cleaned_inputs['Cholesterol'])} mg/dl",
            "FastingBS": "> 120 mg/dl (Diabetic / Pre-diabetic)" if cleaned_inputs["FastingBS"] == 1 else "<= 120 mg/dl (Normal)",
            "RestingECG": {
                0: "Normal Sinus (0)",
                1: "ST-T Wave Abnormality (1)",
                2: "Left Ventricular Hypertrophy (2)"
            }.get(cleaned_inputs["RestingECG"], str(cleaned_inputs["RestingECG"])),
            "MaxHR": f"{int(cleaned_inputs['MaxHR'])} bpm",
            "ExerciseAngina": "Yes (Triggered by exertion)" if cleaned_inputs["ExerciseAngina"] == 1 else "No (Absent during stress)",
            "Oldpeak": f"{cleaned_inputs['Oldpeak']:.1f} mm ST depression",
            "ST_Slope": {
                1: "Upsloping (Normal)",
                2: "Flat (Ischemic)",
                3: "Downsloping (Severe)"
            }.get(cleaned_inputs["ST_Slope"], str(cleaned_inputs["ST_Slope"])),
            "CA": f"{cleaned_inputs['CA']} major vessels",
            "Thal": {
                3: "Normal Perfusion (3)",
                6: "Fixed Defect / Scar (6)",
                7: "Reversible Defect / Ischemia (7)"
            }.get(cleaned_inputs["Thal"], str(cleaned_inputs["Thal"]))
        }

        return {
            "prediction": predicted_class,
            "prediction_label": diagnosis_label,
            "probability_disease": round(prob_disease, 4),
            "probability_healthy": round(prob_healthy, 4),
            "probability_disease_pct": round(prob_disease * 100.0, 1),
            "probability_healthy_pct": round(prob_healthy * 100.0, 1),
            "confidence_pct": round(confidence_pct, 1),
            "risk_level": risk_level,
            "risk_badge_class": risk_badge_class,
            "clinical_recommendation": recommendation,
            "patient_details": formatted_inputs
        }


# Convenience module-level predictor function
def make_prediction(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute prediction using the singleton predictor instance.

    Args:
        patient_data (Dict[str, Any]): Dictionary of 13 clinical features.

    Returns:
        Dict[str, Any]: Evaluated diagnostic outcome dictionary.
    """
    predictor = CardioPredictor.get_instance()
    return predictor.predict_patient(patient_data)


if __name__ == "__main__":
    print("Testing pipeline/predict.py with mock patient cases...")

    # Mock Patient A: Healthy 45-year-old female
    healthy_case = {
        "Age": 45, "Sex": 0, "ChestPainType": 2, "RestingBP": 115, "Cholesterol": 190,
        "FastingBS": 0, "RestingECG": 0, "MaxHR": 175, "ExerciseAngina": 0,
        "Oldpeak": 0.0, "ST_Slope": 1, "CA": 0, "Thal": 3
    }
    res_a = make_prediction(healthy_case)
    print(f"\nPatient A: {res_a['prediction_label']} ({res_a['risk_level']}, Confidence: {res_a['confidence_pct']}%)")

    # Mock Patient B: High-risk 65-year-old male
    severe_case = {
        "Age": 65, "Sex": 1, "ChestPainType": 4, "RestingBP": 160, "Cholesterol": 286,
        "FastingBS": 1, "RestingECG": 2, "MaxHR": 108, "ExerciseAngina": 1,
        "Oldpeak": 3.2, "ST_Slope": 2, "CA": 2, "Thal": 7
    }
    res_b = make_prediction(severe_case)
    print(f"Patient B: {res_b['prediction_label']} ({res_b['risk_level']}, Confidence: {res_b['confidence_pct']}%)")
    print("\n✅ pipeline/predict.py is fully operational.")
