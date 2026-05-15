import random
import csv
import os
import sys

def generate_diabetes_data(n_rows, output_path, seed=42):
    """Generate a synthetic Pima Indians Diabetes-style dataset."""
    random.seed(seed)

    header = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    rows = []
    for _ in range(n_rows):

        pregnancies = random.randint(0, 15)
        glucose = max(0, int(random.gauss(120, 30)))  # 0 allowed (missing)
        bp = max(0, int(random.gauss(70, 12)))
        skin = max(0, int(random.gauss(25, 10)))
        insulin = max(0, int(random.gauss(100, 80)))
        bmi = round(max(0, random.gauss(32, 7)), 1)
        dpf = round(random.uniform(0.1, 2.5), 3)
        age = random.randint(18, 80)

        # Diabetes probability model (simple rule-based)
        diabetes_prob = 0.10

        if glucose > 140:
            diabetes_prob += 0.25
        if bmi > 35:
            diabetes_prob += 0.15
        if age > 50:
            diabetes_prob += 0.10
        if pregnancies >= 5:
            diabetes_prob += 0.05
        if dpf > 1.0:
            diabetes_prob += 0.10

        outcome = 1 if random.random() < diabetes_prob else 0

        row = [
            pregnancies, glucose, bp, skin,
            insulin, bmi, dpf, age, outcome
        ]
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    diabetes_count = sum(r[-1] for r in rows)
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Diabetes rate: {diabetes_count/n_rows:.1%} ({diabetes_count} positive cases)")


if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/diabetes.csv"
    generate_diabetes_data(n_rows, output_path)
