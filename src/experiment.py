import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from preprocessing import DiabetesPreprocessor
from config_loader import load_config


def run_experiment(params):
    """
    Runs a single experiment using the provided model parameters.
    Returns the accuracy score.
    """
    config = load_config()

    # Load dataset
    df = pd.read_csv(config["data"]["raw_path"])

    # Preprocess
    prep = DiabetesPreprocessor()
    df = prep.run(df)

    # Split features/target
    X = df.drop(columns=[config["data"]["target_column"]])
    y = df[config["data"]["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["preprocessing"]["random_state"],
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Run experiment
    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # Log parameters + metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", score)

        print(f"Experiment completed. Accuracy: {score}")

        return score


# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    default_params = config["model"]["params"]

    print("Running experiment with default model parameters...")
    score = run_experiment(default_params)
    print(f"Final Accuracy: {score}")
