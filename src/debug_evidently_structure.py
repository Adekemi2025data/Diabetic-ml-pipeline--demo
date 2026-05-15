import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("reference_diabetes.csv")
current = pd.read_csv("month3_diabetes.csv")

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference, current_data=current)

result = snapshot.dict()

import json
print(json.dumps(result, indent=2))
