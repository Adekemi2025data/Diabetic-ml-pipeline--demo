import yaml
import os

def load_config(path="config/config.yaml"):
    """Load YAML configuration file reliably in GitHub Actions and locally."""
    # Find the project root by going up until we find the config folder
    current = os.path.abspath(os.path.dirname(__file__))

    while True:
        candidate = os.path.join(current, path)
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(f"Could not find {path}")
        current = parent

