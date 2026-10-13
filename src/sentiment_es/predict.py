"""Predicción rápida desde la terminal."""

import argparse

from .pipeline import build_pipeline, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Texto en español para clasificar")
    parser.add_argument("--data", default="data/sentiment_es.csv")
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    model = build_pipeline()
    model.fit(texts, labels)
    prediction = model.predict([args.text])[0]
    confidence = max(model.predict_proba([args.text])[0])
    print(f"sentiment={prediction} confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
