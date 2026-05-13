import pandas as pd

class DiabetesPreprocessor:

    def validate(self, df):
        """Check for missing values."""
        if df.isnull().sum().sum() > 0:
            print("Warning: dataset contains missing values")

    def run(self, df):
        """Clean the dataset by filling missing values with median."""
        df = df.copy()
        df = df.fillna(df.median())
        return df

    def quality_report(self, df):
        """Return basic data quality metrics."""
        return {
            "total_nulls": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }
