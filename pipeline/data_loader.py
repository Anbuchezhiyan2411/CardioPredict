"""Data Ingestion & Loading Module.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : pipeline/data_loader.py
Purpose : Ingest the cleaned dataset (Phase 2 output), validate its integrity,
          and split it into Feature matrix (X) and Target vector (y).
=============================================================================
"""

import os
from typing import Optional, Tuple
import pandas as pd

try:
    from pipeline.utils import get_project_root, print_section_header
except ImportError:
    from utils import get_project_root, print_section_header


def load_cleaned_data(
    file_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed and cleaned UCI Cleveland Heart Disease dataset.

    Args:
        file_path (Optional[str]): Explicit path to cleaned CSV. If None,
            defaults to 'dataset/heart_cleaned.csv' or 'dataset/cleaned_cardiopredict.csv'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X (pd.DataFrame): 13 clinical predictor features.
            - y (pd.Series): Binary target vector (0: Healthy, 1: Heart Disease).

    Raises:
        FileNotFoundError: If the cleaned dataset cannot be found.
        ValueError: If unexpected missing values or schema mismatch are detected.
    """
    if file_path is None:
        root_dir = get_project_root()
        # Check standard cleaned dataset paths
        candidate_1 = os.path.join(root_dir, "dataset", "heart_cleaned.csv")
        candidate_2 = os.path.join(root_dir, "dataset", "cleaned_cardiopredict.csv")

        if os.path.exists(candidate_1):
            file_path = candidate_1
        elif os.path.exists(candidate_2):
            file_path = candidate_2
        else:
            raise FileNotFoundError(
                "Cleaned dataset not found! Please run Phase 2 data cleaning first."
            )

    print(f"📖 Ingesting cleaned dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # 1. Integrity check: Verify zero missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        raise ValueError(f"Integrity Error: Dataset contains {missing_count} unhandled missing values.")

    # 2. Schema check: Verify presence of Target column
    if "Target" not in df.columns:
        raise ValueError("Schema Error: 'Target' column not present in the dataset.")

    # 3. Separate features (X) and target (y)
    feature_matrix = df.drop(columns=["Target"])
    target_vector = df["Target"]

    print(f"✅ Ingestion complete: {feature_matrix.shape[0]} patients, {feature_matrix.shape[1]} features.")
    print(f"   Target distribution: Healthy (0) = {(target_vector == 0).sum()}, "
          f"Heart Disease (1) = {(target_vector == 1).sum()}")

    return feature_matrix, target_vector


if __name__ == "__main__":
    print_section_header("Data Loader Module Verification")
    X, y = load_cleaned_data()
    print("\nFeature Matrix Head:")
    print(X.head(3))
    print("\nTarget Vector Value Counts:")
    print(y.value_counts())
    print("\n✅ pipeline/data_loader.py is fully operational.")
