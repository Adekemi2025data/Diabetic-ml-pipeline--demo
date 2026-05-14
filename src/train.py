import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing import DiabetesPreprocessor
from config_loader import load_config
import mlflow


def load_data(config):
    """Load dataset from local path or URL."""
    source = config["data"]["source"]

    print(f"Loading diabetes dataset from {source}...")
    df = pd.read_csv(source)

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """Apply preprocessing pipeline."""
    prep = DiabetesPreprocessor()
    df_clean = prep.run(df)
    quality = prep.quality_report(df_clean)

    print(f"Data quality: {quality['total_nulls']} nulls, {quality['total_rows']} rows")
    return df_clean, quality


def split_data(df, config):
    """Split into train/test sets."""
    target = config["data"]["target"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(config):
    """Train RandomForest model and log with MLflow."""
    df = load_data(config)
    df_clean, quality = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df_clean, config)

    # FIXED: use "parameters" (not "params")
    params = config["model"]["parameters"]
    model = RandomForestClassifier(**params)

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model accuracy: {acc:.4f}")

    # MLflow logging
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("total_nulls", quality["total_nulls"])
        mlflow.log_metric("total_rows", quality["total_rows"])

        # Save model
        model_path = config["training"]["output_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

    return {
        "accuracy": acc,
        "quality": quality,
        "model_path": model_path
    }


if __name__ == "__main__":
    config = load_config()
    metrics = train_model(config)
    print("Training complete.")
    print(metrics)
