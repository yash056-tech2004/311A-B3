import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
MODEL_PATH = ROOT / "model" / "iris_rf.pkl"
METRICS_PATH = ROOT / "results" / "metrics.json"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_iris_hash(data: np.ndarray, target: np.ndarray) -> str:
    payload = np.concatenate(
        (data.astype(np.float64).ravel(), target.astype(np.int64).ravel())
    ).tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    config = load_config(CONFIG_PATH)
    experiment_config = config["yamlexperiment"]
    model_config = config["model"]

    seed = experiment_config["random_seed"]
    set_random_seeds(seed)

    iris = load_iris()
    data_hash = compute_iris_hash(iris.data, iris.target)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=model_config["test_size"],
        random_state=seed,
        stratify=iris.target,
    )

    if model_config["type"] != "RandomForest":
        raise ValueError(f'Unsupported model type: {model_config["type"]}')

    classifier = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=seed,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Experiment: {experiment_config['name']}")
    print(f"Dataset SHA-256: {data_hash}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(classifier, file)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "seed": seed,
        "data_hash": data_hash,
        "accuracy": accuracy,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
