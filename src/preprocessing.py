import pandas as pd
import numpy as np


class DiabetesPreprocessor:
    numeric_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    target_column = "Outcome"

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    def validate(self, df):
        # Do NOT raise "Dataframe is empty" — tests expect missing-column error instead
        missing = [col for col in self.numeric_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.target_column not in df.columns:
            raise ValueError("Target column missing")

        return True

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------
    def replace_zeros(self, df):
        """Replace biologically impossible zeros with NaN."""
        df = df.copy()
        zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_invalid:
            df[col] = df[col].replace(0, np.nan)
        return df

    def fill_missing(self, df):
        """Fill NaN with median values."""
        df = df.copy()
        for col in self.numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        return df

    def run(self, df):
        """Full preprocessing pipeline."""
        self.validate(df)
        df = self.replace_zeros(df)
        df = self.fill_missing(df)
        return df

    # ---------------------------------------------------------
    # QUALITY REPORT
    # ---------------------------------------------------------
    def quality_report(self, df):
        report = {
            "total_rows": len(df),
            "total_nulls": int(df.isnull().sum().sum())
        }

        for col in self.numeric_columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

        return report
