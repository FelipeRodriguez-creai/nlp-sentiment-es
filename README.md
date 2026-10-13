# NLP Sentiment ES

Clasificador reproducible de sentimiento para textos cortos en español.

El proyecto entrena un pipeline de `TF-IDF` + regresión logística sobre un conjunto pequeño y transparente de ejemplos etiquetados. Está pensado como una base educativa y extensible para experimentos de NLP, no como un modelo listo para decisiones de alto impacto.

## Características

- Entrenamiento determinista con semilla configurable.
- Preprocesamiento conservador que mantiene acentos y palabras útiles.
- Métricas de accuracy, precision, recall y F1.
- Inferencia desde la línea de comandos.
- Pruebas unitarias sin descargar datos externos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Uso

Entrenar y evaluar:

```bash
python -m sentiment_es.train --data data/sentiment_es.csv
```

Predecir una frase:

```bash
python -m sentiment_es.predict "El servicio fue rápido y excelente"
```

## Reproducibilidad

El entrenamiento usa la semilla `42` por defecto y validación cruzada estratificada de tres particiones. Cambia la semilla con `--seed` para comparar experimentos de forma controlada.

## Limitaciones

El dataset incluido es pequeño y balanceado para facilitar la reproducibilidad. Para uso real se necesita un corpus mayor, revisión de sesgos, validación temporal y evaluación por dominio.

## Licencia

MIT
