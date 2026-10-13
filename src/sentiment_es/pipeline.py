"""Funciones puras y pequeñas para entrenar el baseline de NLP."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    """Carga textos y etiquetas, validando el esquema mínimo esperado."""
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError("El dataset contiene valores nulos en text o label")
    return frame["text"].astype(str).tolist(), frame["label"].astype(str).tolist()


def build_pipeline(seed: int = 42) -> Pipeline:
    """Construye un pipeline determinista para clasificación binaria."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )
