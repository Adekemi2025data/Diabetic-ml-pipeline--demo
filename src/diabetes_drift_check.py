import pandas as pd
import sys
import json
import os
from evidently import Report
from evidently.presets import DataDriftPreset

# Thresholds
DRIFT_SHARE_WARNING = 0.20
DRIFT_SHARE_CRITICAL = 0.40


def check_drift(reference_path, current_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    result = snapshot.dict()

    metrics = result["metrics"]

    # --- OVERALL DRIFT SUMMARY ---
    drift_summary = metrics[0]["value"]
    drifted_count = int(drift_summary["count"])
    drift_share = float(drift_summary["share"])

    # Total features = all ValueDrift metrics
    total_features = len(metrics) - 1

    # --- PER-FEATURE DRIFT ---
    drifted_features = []

    for metric in metrics[1:]:
        column = metric["config"]["column"]
        p_value = float(metric["value"])
        threshold = metric["config"]["threshold"]

        if p_value < threshold:
            drifted_features.append(column)

    # --- BUILD RESULT ---
    status = "ok"
    if drift_share >= DRIFT_SHARE_CRITICAL:
        status = "critical"
    elif drift_share >= DRIFT_SHARE_WARNING:
        status = "warning"

    return {
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "status": status,
        "drifted_feature_names": drifted_features,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diabetes_drift_check.py <reference.csv> <current.csv>")
        sys.exit(1)

    reference_path = sys.argv[1]
    current_path = sys.argv[2]

    os.makedirs("reports", exist_ok=True)

    print(f"Checking drift: {current_path} vs {reference_path}")
    print("=" * 60)

    result = check_drift(reference_path, current_path)

    print(f"Features drifted: {result['drifted_features']}/{result['total_features']} "
          f"({result['drift_share']*100:.1f}%)")
    print(f"Status: {result['status'].upper()}")

    if result["drifted_feature_names"]:
        print("\nDrifted features:")
        for f in result["drifted_feature_names"]:
            print(f"  - {f}")

    # Save JSON
    with open("reports/drift_check_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved to reports/drift_check_result.json")

    # Exit codes
    if result["status"] == "critical":
        sys.exit(1)
    else:
        sys.exit(0)
