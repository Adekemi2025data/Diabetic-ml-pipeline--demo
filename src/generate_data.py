import random
import csv
import os
import sys

def generate_diabetes_data(n_rows, output_path, seed=42):
    """Generate a synthetic diabetes dataset similar to Pima Indians Diabetes."""
    random.seed(seed)

    header = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    rows = []
    for _ in range(n_rows):

        pregnancies = random.randint(0, 15)
        glucose = random.randint(60, 200)
        blood_pressure = random.randint(40, 120)
        skin_thickness = random.randint(5, 50)
        insulin = random.randint(15, 300)
        bmi = round(random.uniform(15.0, 50.0), 1)
        dpf = round(random.uniform(0.1, 2.5), 3)
        age = random.randint(18, 80)

        # Base probability
        diabetes_prob = 0.10

        # Increase probability based on risk factors
        if glucose > 140:
            diabetes_prob += 0.25
        if bmi > 30:
            diabetes_prob += 0.15
        if age > 50:
            diabetes_prob += 0.10
        if insulin > 200:
            diabetes_prob += 0.10
        if dpf > 1.0:
            diabetes_prob += 0.10

        outcome = 1 if random.random() < diabetes_prob else 0

        row = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, outcome
        ]
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    positive_cases = sum(r[-1] for r in rows)
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Diabetes rate: {positive_cases/n_rows:.1%} ({positive_cases} positive cases)")

if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/diabetes.csv"
    generate_diabetes_data(n_rows, output_path)
