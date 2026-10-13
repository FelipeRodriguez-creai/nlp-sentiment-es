from pathlib import Path

import pytest

from sentiment_es.pipeline import build_pipeline, load_dataset


DATA = Path(__file__).parents[1] / "data" / "sentiment_es.csv"


def test_load_dataset_has_texts_and_labels() -> None:
    texts, labels = load_dataset(DATA)
    assert len(texts) == len(labels) == 36
    assert set(labels) == {"positive", "negative"}


def test_pipeline_predicts_known_examples() -> None:
    texts, labels = load_dataset(DATA)
    model = build_pipeline()
    model.fit(texts, labels)
    predictions = model.predict(["La atención fue excelente", "El pedido llegó tarde"])
    assert list(predictions) == ["positive", "negative"]


def test_load_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("message,label\nHola,positive\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_dataset(invalid)
