"""Entrenamiento y evaluación del baseline."""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate

from .pipeline import build_pipeline, load_dataset


def train_and_evaluate(data_path: str, seed: int = 42) -> dict[str, float]:
    texts, labels = load_dataset(data_path)
    model = build_pipeline(seed)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results = cross_validate(
        model,
        texts,
        labels,
        cv=folds,
        scoring=("accuracy", "precision_macro", "recall_macro", "f1_macro"),
    )
    return {
        "accuracy": results["test_accuracy"].mean(),
        "precision_macro": results["test_precision_macro"].mean(),
        "recall_macro": results["test_recall_macro"].mean(),
        "f1_macro": results["test_f1_macro"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sentiment_es.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.data, args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
