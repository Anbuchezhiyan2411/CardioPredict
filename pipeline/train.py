"""Model Training & Serialisation Module.

=============================================================================
CardioPredict: Heart Disease Prediction and Analytics using Machine Learning
=============================================================================
Author  : Senior Data Scientist & Python Mentor
Date    : November 2026
Module  : pipeline/train.py
Purpose : Train five distinct classification algorithms on the standardized
          training cohort, compute train/test accuracy scores, document learning
          observations, and save model artifacts inside models/ for Flask.
=============================================================================
"""

import os
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

try:
    from pipeline.preprocessing import prepare_datasets
    from pipeline.utils import print_section_header, save_artifact
except ImportError:
    from preprocessing import prepare_datasets
    from utils import print_section_header, save_artifact


def explain_logistic_regression() -> str:
    """Return theoretical and mathematical intuition for Logistic Regression."""
    return (
        "1. LOGISTIC REGRESSION\n"
        "-----------------------------------------------------------------------\n"
        "• Mathematical Intuition:\n"
        "  Logistic Regression models the log-odds of the positive class as a linear\n"
        "  combination of input predictors:\n"
        "      ln( p / (1 - p) ) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ\n"
        "  The Sigmoid (logistic) activation function maps linear outputs to probabilities:\n"
        "      σ(z) = 1 / (1 + e^(-z))  where z ∈ (-∞, +∞) and σ(z) ∈ [0, 1]\n"
        "  Parameters are estimated by maximizing the log-likelihood (minimizing binary cross-entropy).\n"
        "• Advantages:\n"
        "  - Highly interpretable: Coefficients directly represent odds ratios.\n"
        "  - Fast computation with minimal memory footprint.\n"
        "  - Provides well-calibrated posterior probabilities, not just binary classifications.\n"
        "• Disadvantages:\n"
        "  - Assumes a linear decision boundary in log-odds space; fails with non-linear interactions.\n"
        "  - Vulnerable to collinearity and requires feature scaling when regularized.\n"
        "• When to Use:\n"
        "  - When explainability is critical (e.g. clinical medicine) and a linear baseline is required."
    )


def explain_decision_tree() -> str:
    """Return theoretical and mathematical intuition for Decision Trees."""
    return (
        "2. DECISION TREE CLASSIFIER\n"
        "-----------------------------------------------------------------------\n"
        "• Mathematical Intuition:\n"
        "  A non-parametric algorithm that recursively partitions the feature space into\n"
        "  orthogonal hyper-rectangles using greedy threshold splits.\n"
        "  - Gini Impurity: Measures probability of incorrect classification at node t:\n"
        "      Gini(t) = 1 - Σ (pᵢ)²\n"
        "  - Entropy & Information Gain: Measures disorder / uncertainty in bits:\n"
        "      Entropy(t) = - Σ pᵢ log₂(pᵢ)\n"
        "      Information Gain = Entropy(Parent) - Σ [ (|Child_j| / |Parent|) * Entropy(Child_j) ]\n"
        "• Tree Splitting Logic:\n"
        "  At every node, the tree scans all features and thresholds, selecting the split that\n"
        "  maximizes Gini purity gain or Information Gain.\n"
        "• Advantages:\n"
        "  - Completely transparent rule structure (mirrors clinical flowchart decisions: IF Age > 55 AND CA > 1).\n"
        "  - Invariant to monotonic scaling; handles non-linear relationships natively.\n"
        "• Disadvantages:\n"
        "  - High variance: Sensitive to small perturbations in training data.\n"
        "  - Prone to deep overfitting without depth constraints (max_depth, min_samples_leaf)."
    )


def explain_random_forest() -> str:
    """Return theoretical and mathematical intuition for Random Forest."""
    return (
        "3. RANDOM FOREST CLASSIFIER\n"
        "-----------------------------------------------------------------------\n"
        "• Mathematical Intuition:\n"
        "  An ensemble meta-estimator that fits an array of de-correlated decision trees\n"
        "  and aggregates their predictions via majority voting.\n"
        "• Bootstrap Aggregation (Bagging):\n"
        "  Each tree is trained on an independent bootstrap sample of size N drawn with\n"
        "  replacement from the training cohort (~63.2% unique samples per tree).\n"
        "• Random Subspace / Feature Selection:\n"
        "  At each candidate split, only a random subset of m ≈ √p features (√13 ≈ 3 or 4)\n"
        "  is evaluated. This de-correlates the individual trees, ensuring a dominant predictor\n"
        "  (like Thal or CA) does not dictate every single tree root.\n"
        "• Advantages:\n"
        "  - Drastically slashes variance without inflating bias.\n"
        "  - Highly resistant to overfitting; provides robust out-of-bag error estimation.\n"
        "  - Naturally captures complex non-linear feature interactions."
    )


def explain_svm() -> str:
    """Return theoretical and mathematical intuition for Support Vector Machines."""
    return (
        "4. SUPPORT VECTOR MACHINE (SVM)\n"
        "-----------------------------------------------------------------------\n"
        "• Mathematical Intuition:\n"
        "  Finds the optimal separating hyperplane that maximizes the geometric margin between\n"
        "  the two classes in feature space:\n"
        "      Hyperplane: wᵀx + b = 0\n"
        "  - Margin: 2 / ||w||. Maximizing the margin minimizes the structural risk bound.\n"
        "  - Support Vectors: The critical data points closest to the hyperplane that dictate its boundary.\n"
        "  - Kernel Trick (RBF Kernel): K(x, x') = exp( -γ ||x - x'||² )\n"
        "    Implicitly projects non-linearly separable clinical data into an infinite-dimensional\n"
        "    Hilbert space where a linear boundary exists, without computing explicit coordinates.\n"
        "• Advantages:\n"
        "  - Highly effective in high-dimensional feature spaces.\n"
        "  - Robust against overfitting when regularized with margin parameter C.\n"
        "• Disadvantages:\n"
        "  - Computationally intensive on very large datasets; strictly requires feature standardization.\n"
        "  - Operates as a 'black box' geometry with limited direct interpretability."
    )


def explain_knn() -> str:
    """Return theoretical and mathematical intuition for K-Nearest Neighbors."""
    return (
        "5. K-NEAREST NEIGHBORS (KNN)\n"
        "-----------------------------------------------------------------------\n"
        "• Mathematical Intuition:\n"
        "  An instance-based, non-parametric, 'lazy' learning algorithm. It stores the training\n"
        "  dataset in memory and computes distance metrics at query time.\n"
        "• Distance Metric (Euclidean Distance):\n"
        "  d(p, q) = √ [ Σᵢ (pᵢ - qᵢ)² ]\n"
        "• Role of Hyperparameter K:\n"
        "  - Small K (e.g. K=1): Low bias, high variance (sensitive to individual noisy outliers).\n"
        "  - Large K (e.g. K=25): High bias, low variance (over-smoothes complex boundaries).\n"
        "  - Optimal K=7: Balances local neighborhood sensitivity with noise resilience.\n"
        "• Advantages:\n"
        "  - Simple, zero training phase assumptions; naturally adapts to complex decision surfaces.\n"
        "• Disadvantages:\n"
        "  - Memory and compute intensive during inference (O(N*d) per test patient).\n"
        "  - Severely impaired if continuous features are not standardized."
    )


def initialize_models() -> Dict[str, Any]:
    """Instantiate the 5 classification models with validated hyperparameters.

    Returns:
        Dict[str, Any]: Dictionary of initialized model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=4,              # Regularized depth to prevent excessive leaf fragmentation
            min_samples_split=6,
            min_samples_leaf=3,
            criterion="gini"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,          # 100 de-correlated estimators
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2
        ),
        "Support Vector Machine": SVC(
            random_state=42,
            kernel="rbf",              # Radial Basis Function for non-linear boundary
            C=1.0,
            gamma="scale"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,             # Odd K to prevent voting ties
            metric="minkowski",
            p=2                        # Euclidean distance metric
        )
    }


def train_and_persist_all_models() -> Dict[str, Dict[str, Any]]:
    """Execute model training on the preprocessed training set, log metrics, and persist.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing models, scores, and observations.
    """
    print_section_header("Phase 7: Machine Learning Model Training")

    # Step 1: Ingest preprocessed training and testing arrays
    X_train, X_test, y_train, y_test, feature_cols = prepare_datasets()

    # Model file mapping dictionary
    model_filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "Support Vector Machine": "svm.pkl",
        "K-Nearest Neighbors": "knn.pkl"
    }

    # Algorithm explanations mapping
    explanations = {
        "Logistic Regression": explain_logistic_regression(),
        "Decision Tree": explain_decision_tree(),
        "Random Forest": explain_random_forest(),
        "Support Vector Machine": explain_svm(),
        "K-Nearest Neighbors": explain_knn()
    }

    # Initialize estimators
    models = initialize_models()
    training_results = {}

    print("\n" + "=" * 75)
    print("BEGINNING INDIVIDUAL MODEL TRAINING & VERIFICATION")
    print("=" * 75)

    for model_name, model_instance in models.items():
        print_section_header(f"Training: {model_name}")
        print(explanations[model_name])

        # 1. Fit estimator on training cohort
        model_instance.fit(X_train, y_train)

        # 2. Predict on train and test subsets
        train_preds = model_instance.predict(X_train)
        test_preds = model_instance.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        # 3. Document diagnostic learning observations
        observations = []
        if train_acc - test_acc > 0.15:
            observations.append("Signs of mild overfitting (train accuracy exceeds test by > 15%).")
        elif abs(train_acc - test_acc) <= 0.05:
            observations.append("Superb generalization balance (train and test accuracies are tightly aligned within 5%).")
        else:
            observations.append("Healthy learning curve with expected minor generalization gap.")

        if test_acc >= 0.85:
            observations.append("Strong clinical predictive benchmark (> 85% accuracy on unseen holdout patients).")
        else:
            observations.append("Moderate predictive capacity on the test cohort.")

        print(f"\n📈 Performance Metrics for {model_name}:")
        print(f"   • Training Accuracy : {train_acc * 100:.2f}% ({int(train_acc * len(y_train))}/{len(y_train)} correct)")
        print(f"   • Testing Accuracy  : {test_acc * 100:.2f}% ({int(test_acc * len(y_test))}/{len(y_test)} correct)")
        print(f"   • Learning Insights : {' '.join(observations)}")

        # 4. Save trained model artifact to models/
        save_filename = model_filename_map[model_name]
        saved_path = save_artifact(model_instance, save_filename)

        training_results[model_name] = {
            "model": model_instance,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "observations": observations,
            "artifact_path": saved_path
        }

    # Master Training Summary Table
    print_section_header("Master Model Training Summary (Phase 7)")
    summary_data = []
    for name, res in training_results.items():
        summary_data.append({
            "Algorithm": name,
            "Training Accuracy (%)": f"{res['train_accuracy'] * 100:.2f}%",
            "Testing Accuracy (%)": f"{res['test_accuracy'] * 100:.2f}%",
            "Holdout Generalization": res['observations'][0],
            "Saved Artifact": model_filename_map[name]
        })

    summary_table = pd.DataFrame(summary_data)
    print(summary_table.to_string(index=False))

    print("""
⚠️ REMINDER (Per Project Guidelines):
We do NOT declare the single 'Best Model' here in Phase 7.
Formal comparative evaluation, ROC-AUC curves, Confusion Matrices, Precision,
Recall, and F1-Scores will be rigorously executed in Phase 8.
""")
    return training_results


if __name__ == "__main__":
    results = train_and_persist_all_models()
    print("✅ pipeline/train.py completed successfully.")
