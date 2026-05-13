from itertools import product
from config_loader import load_config
from experiment import run_experiment


def sweep():
    # Load YAML config
    config = load_config()

    # Sweep parameter grid from config.yaml
    sweep_cfg = config["sweep"]["parameters"]

    keys = sweep_cfg.keys()
    values = sweep_cfg.values()

    # Cartesian product of all hyperparameter combinations
    for combo in product(*values):
        params = dict(zip(keys, combo))
        print(f"\nRunning sweep with params: {params}")

        score = run_experiment(params)
        print(f"Score: {score}")


if __name__ == "__main__":
    sweep()
