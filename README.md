# Desafíos de Procesamiento de Lenguaje Natural

Este repositorio reúne el desarrollo de cuatro desafíos prácticos de Procesamiento de Lenguaje Natural, implementados en notebooks de Jupyter.

Cada desafío aborda una tarea distinta dentro del pipeline de PLN moderno: desde clasificación de texto y embeddings, hasta modelos generativos y traducción automática con arquitecturas seq2seq.

---

## Estructura del repositorio

- `Desafio_1.ipynb` – Vectorización de texto y clasificación con Naïve Bayes sobre el dataset **20 Newsgroups**.
- `Desafio_2.ipynb` – Entrenamiento de **Word2Vec** tipo Skip-gram sobre letras de canciones de Nirvana y análisis de similitudes semánticas.
- `Desafio_3.ipynb` – Modelo de lenguaje **a nivel carácter** basado en SimpleRNN sobre el libro *Así habló Zaratustra*, incluyendo cálculo de **perplejidad**.
- `Desafio_4.ipynb` – Traductor **inglés → español** con arquitectura **LSTM encoder–decoder** y embeddings pre-entrenados (GloVe).



---

## Requisitos

Los desafíos utilizan principalmente:

- Python 3.x
- Manejo de datos y utilidades:
  - `numpy`
  - `pandas`
- Visualización:
  - `matplotlib`
  - `seaborn`
- Modelado clásico de ML:
  - `scikit-learn`
- PLN y embeddings:
  - `gensim`
  - `bs4` (BeautifulSoup) + `lxml`
- Deep Learning:
  - `tensorflow` / `keras`
- Utilidades varias:
  - `gdown` (descarga de embeddings)
  - `urllib` / `requests`
  - `multiprocessing` (cuando aplica)
  - `os`, `pathlib`, `pickle`, etc.



Ejemplo de instalación genérica:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
