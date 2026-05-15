import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
import os

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Load the diabetes drift simulation outputs
reference = pd.read_csv("reference_diabetes.csv")
month3 = pd.read_csv("month3_diabetes.csv")

# Only monitor the most medically important features
critical_features = [
    "Glucose",
    "BloodPressure",
    "BMI",
    "Age",
    "SkinThickness"
]

# Build a report with individual column drift metrics
metrics = [ValueDrift(column=col) for col in critical_features]

report = Report(metrics=metrics)
snapshot = report.run(reference_data=reference, current_data=month3)

result = snapshot.dict()

print("Critical Feature Drift Analysis (Month 3)")
print("=" * 60)

for i, feature in enumerate(critical_features):
    metric = result["metrics"][i]
    score = float(metric["value"])
    threshold = metric["config"]["threshold"]
    method = metric["config"]["method"]
    drifted = score >= threshold

    status = "DRIFT DETECTED" if drifted else "stable"
    print(f"\n{feature}:")
    print(f"  Status:    {status}")
    print(f"  Score:     {score:.6f}")
    print(f"  Threshold: {threshold}")
    print(f"  Test used: {method}")

# Save HTML report
output_path = "reports/critical_features_month3.html"
snapshot.save_html(output_path)
print(f"\nDetailed report saved to: {output_path}")

