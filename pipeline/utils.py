"""Utility Functions Module.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : pipeline/utils.py
Purpose : Common helper functions for serializing and deserializing models,
          scalers, and column schemas, as well as console formatting.
=============================================================================
"""

import os
from typing import Any
import joblib


def get_project_root() -> str:
    """Return the absolute path to the project root directory.

    Returns:
        str: Absolute path to CardioPredict root folder.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_artifact(artifact: Any, filename: str, subfolder: str = "models") -> str:
    """Serialize a Python object (model, scaler, schema) using Joblib.

    Args:
        artifact (Any): Object to be saved.
        filename (str): Name of the target file (e.g. 'scaler.pkl').
        subfolder (str): Directory relative to root where artifact will be saved.

    Returns:
        str: Absolute path where artifact was saved.
    """
    root_dir = get_project_root()
    target_dir = os.path.join(root_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)
    joblib.dump(artifact, target_path)
    print(f"💾 Artifact successfully saved: {target_path}")
    return target_path


def load_artifact(filename: str, subfolder: str = "models") -> Any:
    """Deserialize a Python object using Joblib.

    Args:
        filename (str): Name of the file to load.
        subfolder (str): Directory relative to root where artifact lives.

    Returns:
        Any: Unpickled Python object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    root_dir = get_project_root()
    target_path = os.path.join(root_dir, subfolder, filename)

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Requested artifact not found at: {target_path}")

    artifact = joblib.load(target_path)
    print(f"📂 Artifact successfully loaded: {target_path}")
    return artifact


def print_section_header(title: str, width: int = 75) -> None:
    """Print an aesthetically formatted section header for console output.

    Args:
        title (str): Title text to display.
        width (int): Character width of the divider lines.
    """
    print("\n" + "=" * width)
    print(f"   {title}")
    print("=" * width)


if __name__ == "__main__":
    print_section_header("Utils Module Verification")
    root = get_project_root()
    print(f"Project root resolved to: {root}")
    print("✅ pipeline/utils.py is fully operational.")
