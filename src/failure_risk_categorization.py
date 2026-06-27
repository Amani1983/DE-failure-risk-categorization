"""
Risk Categorization and Threshold-Based Failure Prediction for Differential Evolution

This script computes ROC-based thresholds for selected landscape features using
Youden's index. The thresholds are used to support failure risk categorization.

Authors:
    Amani Saad
    Andries P. Engelbrecht
    Salman A. Khan
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

INPUT_FILE = Path("data/Risk_Analysis.xlsx")
OUTPUT_FILE = Path("results/ROC_Threshold_Output.txt")

FEATURES = [
    "FEM_001",
    "FEM_01",
    "FDC",
    "DM_INV",
    "FCI",
    "G_avg",
    "G_dev",
    "FCI_DEV_INV",
]

QMETRIC_COLUMN = "QMetric"
SUCCESS_THRESHOLD = 0.9


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def compute_roc_threshold(feature_values, binary_labels):
    """
    Compute the ROC-based threshold using Youden's index.

    Parameters
    ----------
    feature_values : array-like
        Values of the landscape feature.
    binary_labels : array-like
        Binary class labels, where 1 indicates the positive class.

    Returns
    -------
    best_threshold : float
        Threshold that maximizes Youden's index.
    auc : float
        Area under the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(binary_labels, feature_values)
    youden_index = tpr - fpr
    best_idx = youden_index.argmax()

    best_threshold = thresholds[best_idx]
    auc = roc_auc_score(binary_labels, feature_values)

    return best_threshold, auc


def main():
    """
    Run the ROC threshold analysis.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place Risk_Analysis.xlsx inside the data/ folder."
        )

    df = pd.read_excel(INPUT_FILE)

    # Remove accidental leading/trailing spaces from column names.
    df.columns = df.columns.str.strip()

    if QMETRIC_COLUMN not in df.columns:
        raise ValueError(f"Required column not found: {QMETRIC_COLUMN}")

    # Define failure as the positive class.
    # Failure = 1 when QMetric is below the success threshold.
    df["Failure"] = df[QMETRIC_COLUMN].apply(
        lambda x: 1 if x < SUCCESS_THRESHOLD else 0
    )

    label_column = "Failure"

    results = []

    for feature in FEATURES:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in the input file.")
            continue

        valid_data = df[[feature, label_column]].dropna()

        threshold, auc = compute_roc_threshold(
            valid_data[feature],
            valid_data[label_column]
        )

        results.append(
            {
                "Feature": feature,
                "Threshold (Youden)": round(threshold, 4),
                "AUC": round(auc, 4),
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("ROC-Based Thresholds for Landscape Features\n")
        f.write("=" * 55 + "\n")
        f.write("Positive class: Failure, where QMetric < 0.9\n")
        f.write("=" * 55 + "\n")

        for row in results:
            f.write(
                f"{row['Feature']:15} | "
                f"Threshold: {row['Threshold (Youden)']:>8} | "
                f"AUC: {row['AUC']}\n"
            )

    print(f"\nThreshold results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
