"""Data Preprocessing & Transformation Module.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : pipeline/preprocessing.py
Purpose : Execute stratified train-test splitting, feature encoding,
          standardization (StandardScaler), and persistence of preprocessing
          artifacts (scaler.pkl, encoder.pkl, feature_columns.pkl) for
          direct downstream consumption by Flask.
=============================================================================
"""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from pipeline.data_loader import load_cleaned_data
    from pipeline.feature_engineering import get_feature_taxonomies
    from pipeline.utils import get_project_root, print_section_header, save_artifact
except ImportError:
    from data_loader import load_cleaned_data
    from feature_engineering import get_feature_taxonomies
    from utils import get_project_root, print_section_header, save_artifact


def explain_encoding_strategy() -> str:
    """Provide detailed explanation of encoding methodologies for clinical data.

    Returns:
        str: Technical and medical justification of encoding choices.
    """
    return (
        "ENCODING STRATEGY & METHODOLOGICAL JUSTIFICATION:\n"
        "-----------------------------------------------------------------------\n"
        "1. Binary Features (Sex, FastingBS, ExerciseAngina):\n"
        "   - These features exist naturally in binary boolean space {0, 1}.\n"
        "   - 0 = Absence / Female / Fasting Sugar <= 120 / No Exercise Angina\n"
        "   - 1 = Presence / Male / Fasting Sugar > 120 / Exercise Angina Present\n"
        "   - Method: Natural binary numeric encoding (already preserved, no extra dummy\n"
        "     variables needed; avoids artificial dimensionality inflation).\n"
        "\n"
        "2. Discrete / Ordinal Features (CA):\n"
        "   - Represents integer count of fluoroscopy-calcified major vessels (0, 1, 2, 3).\n"
        "   - Has genuine arithmetic order (3 blocked vessels is worse than 1).\n"
        "   - Method: Direct integer encoding preserves natural clinical monotonicity.\n"
        "\n"
        "3. Nominal Multi-Class Features (ChestPainType, RestingECG, ST_Slope, Thal):\n"
        "   - In medical decision support, tree ensembles (Random Forest, Decision Tree)\n"
        "     readily partition integer-coded categorical clinical tiers,\n"
        "     while linear and distance models (Logistic Regression, SVM, KNN)\n"
        "     operate efficiently once standard-scaled.\n"
        "   - An encoder dictionary mapping is serialized to models/encoder.pkl to guarantee\n"
        "     bidirectional category mapping for Flask form inputs."
    )


def explain_feature_scaling() -> str:
    """Explain the mathematical necessity of StandardScaler and algorithm sensitivity.

    Returns:
        str: Comprehensive explanation of feature scaling in machine learning.
    """
    return (
        "FEATURE SCALING JUSTIFICATION (StandardScaler):\n"
        "-----------------------------------------------------------------------\n"
        "1. Mathematical Formulation:\n"
        "   z = (x - mean) / std\n"
        "   Centers every feature at mean = 0 with unit variance (std = 1).\n"
        "\n"
        "2. Why Scaling Is Strictly Necessary:\n"
        "   In our dataset, Cholesterol ranges up to 564 mg/dl and MaxHR up to 202 bpm,\n"
        "   while Oldpeak is measured in decimals (0.0 to 6.2 mm) and Sex is 0 or 1.\n"
        "   Without scaling, high-magnitude features disproportionately dominate distance\n"
        "   metrics and gradient updates.\n"
        "\n"
        "3. Algorithm Sensitivity Breakdown:\n"
        "   • K-Nearest Neighbors (KNN): HIGHLY SENSITIVE. Relies on Euclidean distance;\n"
        "     unscaled features will render small-range features completely irrelevant.\n"
        "   • Support Vector Machine (SVM): HIGHLY SENSITIVE. Maximizes margin based on dot\n"
        "     products; unscaled features distort the hyperplane geometry.\n"
        "   • Logistic Regression: HIGHLY SENSITIVE. Gradient descent converges much faster,\n"
        "     and L2 regularization penalizes weights equally across all features.\n"
        "   • Decision Tree & Random Forest: INVARIANT. Tree splits rely on monotonic rank\n"
        "     thresholds (e.g. x <= 130), so scale does not change split points.\n"
        "\n"
        "4. Prevention of Data Leakage:\n"
        "   StandardScaler is fitted EXCLUSIVELY on X_train. The learned mean and std are\n"
        "   then applied to transform X_test and later the Flask incoming web inputs."
    )


def explain_train_test_split() -> str:
    """Explain the rationale behind 80/20 train-test splitting and stratification.

    Returns:
        str: Justification for stratified train-test splitting.
    """
    return (
        "TRAIN-TEST SPLIT METHODOLOGY (80/20 Stratified Split, Random State = 42):\n"
        "-----------------------------------------------------------------------\n"
        "1. Partition Ratio (80% Train, 20% Test):\n"
        "   - 80% (242 patients): Provides sufficient statistical power for model\n"
        "     convergence across 13 parameters.\n"
        "   - 20% (61 patients): A gold-standard independent hold-out test set to evaluate\n"
        "     unbiased generalization error.\n"
        "\n"
        "2. Why Stratification (stratify=y) Is Essential:\n"
        "   - Total cohort has 54.1% healthy and 45.9% diseased patients.\n"
        "   - Stratified sampling guarantees that both training and testing subsets\n"
        "     preserve the exact same 54:46 ratio, eliminating sampling bias.\n"
        "\n"
        "3. Reproducibility (random_state=42):\n"
        "   - Fixes the pseudo-random generator seed, ensuring all college reviewers\n"
        "     and evaluators replicate the identical data splits and metrics."
    )


def prepare_datasets(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
    """Complete preprocessing pipeline: split data, scale features, and persist artifacts.

    Args:
        test_size (float): Proportion of dataset to hold out for testing (default: 0.20).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple containing:
            - X_train_scaled (np.ndarray): Scaled training features.
            - X_test_scaled (np.ndarray): Scaled testing features.
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Testing labels.
            - feature_columns (List[str]): List of predictor column names.
    """
    print_section_header("Executing Phase 6: Preprocessing & Pre-Training Pipeline")

    # Step 1 & 2: Ingest clean data and separate X and y
    X, y = load_cleaned_data()
    feature_columns = list(X.columns)

    print(f"\nFeature Columns ({len(feature_columns)} attributes):")
    for idx, col in enumerate(feature_columns, 1):
        print(f"  {idx:2d}. {col}")

    # Step 8: Stratified Train-Test Split (80/20)
    print("\n" + explain_train_test_split())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nSplit Execution Results:")
    print(f"  • Training Samples  : {X_train.shape[0]} patients (80%)")
    print(f"  • Testing Samples   : {X_test.shape[0]} patients (20%)")
    print(f"  • Train Class Ratio : Healthy={(y_train == 0).sum()} ({((y_train == 0).sum()/len(y_train))*100:.1f}%), "
          f"Disease={(y_train == 1).sum()} ({((y_train == 1).sum()/len(y_train))*100:.1f}%)")
    print(f"  • Test Class Ratio  : Healthy={(y_test == 0).sum()} ({((y_test == 0).sum()/len(y_test))*100:.1f}%), "
          f"Disease={(y_test == 1).sum()} ({((y_test == 1).sum()/len(y_test))*100:.1f}%)")

    # Step 4: Encoding Documentation & Metadata persistence
    print("\n" + explain_encoding_strategy())
    encoder_metadata = {
        "binary_features": ["Sex", "FastingBS", "ExerciseAngina"],
        "discrete_features": ["CA"],
        "categorical_features": {
            "ChestPainType": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"},
            "RestingECG": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
            "ST_Slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
            "Thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
        }
    }

    # Step 5: Feature Scaling via StandardScaler
    print("\n" + explain_feature_scaling())
    scaler = StandardScaler()

    # Fit scaler strictly on training set to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nScaling Complete:")
    print(f"  • X_train_scaled shape: {X_train_scaled.shape}")
    print(f"  • X_test_scaled shape : {X_test_scaled.shape}")
    print(f"  • Feature Mean after scaling: {np.mean(X_train_scaled, axis=0).round(2)}")
    print(f"  • Feature Std after scaling : {np.std(X_train_scaled, axis=0).round(2)}")

    # Step 9: Save preprocessing artifacts into models/
    print_section_header("Step 9: Persisting Preprocessing Artifacts for Flask")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(encoder_metadata, "encoder.pkl")
    save_artifact(feature_columns, "feature_columns.pkl")

    print("""
WHY FLASK REQUIRES THESE FILES:
1. scaler.pkl: When a user enters patient data into the Flask web form,
   raw inputs (e.g. Cholesterol=240, Age=58) must be normalized using the
   EXACT same mean and variance learned during model training.
2. encoder.pkl: Supplies verified valid categorical bounds and metadata
   for dropdown selection and backend validation.
3. feature_columns.pkl: Guarantees that form fields are fed to the model in
   the exact sequential order expected by the scikit-learn estimators.
""")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns


def preprocess_single_patient(
    patient_dict: Dict[str, float],
    scaler_path: Optional[str] = None,
    feature_cols_path: Optional[str] = None
) -> np.ndarray:
    """Preprocess a single patient record submitted via the Flask web interface.

    Args:
        patient_dict (Dict[str, float]): Dictionary containing 13 clinical features.
        scaler_path (Optional[str]): Path to scaler.pkl. Defaults to standard models directory.
        feature_cols_path (Optional[str]): Path to feature_columns.pkl. Defaults to standard.

    Returns:
        np.ndarray: Scaled 2D array of shape (1, 13) ready for model.predict().
    """
    root_dir = get_project_root()
    if scaler_path is None:
        scaler_path = os.path.join(root_dir, "models", "scaler.pkl")
    if feature_cols_path is None:
        feature_cols_path = os.path.join(root_dir, "models", "feature_columns.pkl")

    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_cols_path)

    # Convert patient dictionary into ordered DataFrame
    patient_df = pd.DataFrame([patient_dict])[feature_columns]

    # Transform using fitted scaler
    patient_scaled = scaler.transform(patient_df)
    return patient_scaled


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, cols = prepare_datasets()
    print("✅ pipeline/preprocessing.py executed successfully.")
