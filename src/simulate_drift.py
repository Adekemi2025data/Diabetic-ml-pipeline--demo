import pandas as pd
import numpy as np

def load_and_prepare():
    """Load the diabetes dataset and do basic cleaning."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    # Replace zeros with NaN for columns where zero is invalid
    numeric_cols = df.columns.drop("Outcome")
    for col in numeric_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def create_reference_and_production(df):
    """
    Split data into reference (training) and production batches.
    Reference = first 60%
    Production = next 40% split into 3 months
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.6)
    reference = df.iloc[:split].copy()
    remaining = df.iloc[split:].copy()

    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:batch_size*2].copy()
    month3 = remaining.iloc[batch_size*2:].copy()

    return reference, month1, month2, month3


def introduce_drift(month2, month3):
    """
    Simulate realistic drift in months 2 and 3 for diabetes dataset.
    Month 1 remains clean.
    """

    # ───────────────────────────────
    # MONTH 2 — Moderate drift
    # ───────────────────────────────

    # Glucose increases slightly (diet changes)
    month2["Glucose"] += np.random.normal(10, 5, len(month2))
    month2["Glucose"] = month2["Glucose"].clip(0, None)

    # BMI increases slightly
    month2["BMI"] += np.random.normal(1.0, 0.5, len(month2))

    # Age distribution shifts slightly older
    month2["Age"] += np.random.normal(1.5, 1.0, len(month2))


    # ───────────────────────────────
    # MONTH 3 — Significant drift
    # ───────────────────────────────

    # Strong glucose drift (population change)
    month3["Glucose"] += np.random.normal(25, 10, len(month3))
    month3["Glucose"] = month3["Glucose"].clip(0, None)

    # Blood pressure drops (new medication trend)
    month3["BloodPressure"] -= np.random.normal(8, 4, len(month3))
    month3["BloodPressure"] = month3["BloodPressure"].clip(0, None)

    # BMI increases more aggressively
    month3["BMI"] += np.random.normal(3.0, 1.0, len(month3))

    # Age shift — more older patients
    older_group = np.random.uniform(50, 80, int(len(month3) * 0.3))
    indices = np.random.choice(month3.index, size=len(older_group), replace=False)
    month3.loc[indices, "Age"] = older_group

    # SkinThickness drift (measurement device change)
    month3["SkinThickness"] *= np.random.uniform(1.2, 1.5)

    return month2, month3


if __name__ == "__main__":
    print("Loading diabetes dataset...")
    df = load_and_prepare()
    print(f"Total rows: {len(df)}")

    print("\nSplitting into reference and production batches...")
    reference, month1, month2, month3 = create_reference_and_production(df)

    print("Introducing drift into months 2 and 3...")
    month2, month3 = introduce_drift(month2, month3)

    print(f"\nReference (training data): {len(reference)} rows")
    print(f"Month 1 (no drift):        {len(month1)} rows")
    print(f"Month 2 (moderate drift):  {len(month2)} rows")
    print(f"Month 3 (significant drift): {len(month3)} rows")

    # Save for drift analysis
    reference.to_csv("reference_diabetes.csv", index=False)
    month1.to_csv("month1_diabetes.csv", index=False)
    month2.to_csv("month2_diabetes.csv", index=False)
    month3.to_csv("month3_diabetes.csv", index=False)

    print("\nData saved. Ready for drift analysis.")
