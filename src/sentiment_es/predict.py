"""Predicción rápida desde la terminal."""

import argparse
import json

from .pipeline import build_pipeline, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Texto en español para clasificar")
    parser.add_argument("--data", default="data/sentiment_es.csv")
    parser.add_argument("--json", action="store_true", help="Imprime una respuesta JSON")
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    model = build_pipeline()
    model.fit(texts, labels)
    prediction = model.predict([args.text])[0]
    confidence = max(model.predict_proba([args.text])[0])
    result = {"text": args.text, "sentiment": prediction, "confidence": round(confidence, 3)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"sentiment={prediction} confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
