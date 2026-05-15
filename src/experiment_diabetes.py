import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ────────────────────────────────────────────────────────────────
# Data Quality Check Function
# ────────────────────────────────────────────────────────────────
def check_data_quality(df, numeric_columns):
    """Return a dictionary of data quality metrics."""
    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report


# ────────────────────────────────────────────────────────────────
# Configuration for Diabetes ML Experiments
# ────────────────────────────────────────────────────────────────
config = {
    "model_type": "logistic_regression",
    "test_size": 0.2,
    "random_state": 42,
    "handle_missing": "median",
    "scale_features": True,
    "features_to_drop": [],

    # Model-specific hyperparameters
    "lr_C": 1.0,
    "rf_n_estimators": 200,
    "rf_max_depth": 8,
    "gb_n_estimators": 100,
    "gb_learning_rate": 0.1,
    "gb_max_depth": 3,
}

# ────────────────────────────────────────────────────────────────
# Load and prepare diabetes dataset
# ────────────────────────────────────────────────────────────────
def load_and_prepare_data(config):
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    print("Loading diabetes dataset...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify numeric columns
    numeric_cols = df.columns.drop("Outcome").tolist()

    # Data quality check BEFORE cleaning
    quality = check_data_quality(df, numeric_cols)
    print(f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates")

    # Drop user-specified columns
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['features_to_drop']}")

    # Handle missing values (zeros = missing in diabetes dataset)
    if config["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
        print("Filled missing values with median (after replacing zeros)")
    elif config["handle_missing"] == "drop":
        before = len(df)
        df = df.replace(0, np.nan).dropna()
        print(f"Dropped rows with missing values: {before} -> {len(df)}")

    # No categorical columns in diabetes dataset
    categorical_cols = []

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y, len(df), numeric_cols, categorical_cols


# ────────────────────────────────────────────────────────────────
# Build model based on config
# ────────────────────────────────────────────────────────────────
def build_model(config):
    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )
    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


# ────────────────────────────────────────────────────────────────
# Run MLflow experiment
# ────────────────────────────────────────────────────────────────
def run_experiment(config):

    mlflow.set_experiment("diabetes-prediction")

    with mlflow.start_run():

        # Log config parameters
        for key, value in config.items():
            mlflow.log_param(key, value)

        # Load data
        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )

        # Scale features
        if config["scale_features"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Train model
        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Get run_id BEFORE registering model
        run_id = mlflow.active_run().info.run_id

        # Register the model in MLflow Model Registry
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="diabetes-predictor"
        )

        # Save config snapshot
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

        # Print results
        print("\n" + "="*50)
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print("="*50)

        print(f"\nMLflow Run ID: {run_id}")
        print("Model registered as: diabetes-predictor")

    return run_id


if __name__ == "__main__":
    run_experiment(config)
