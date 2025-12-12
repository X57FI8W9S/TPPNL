# Desafíos de Procesamiento de Lenguaje Natural

Este repositorio reúne el desarrollo de cuatro desafíos prácticos de Procesamiento de Lenguaje Natural, implementados en notebooks de Jupyter. 



## Desafío 1 – Clasificación de Texto (20 Newsgroups)


<!-- ![Desafío 1](images/desafio1.png) -->

**Objetivo.** Construir y analizar un clasificador de noticias utilizando el dataset **20 Newsgroups**, empleando TF-IDF y modelos Naïve Bayes, además de explorar similitudes entre documentos.  

**Notebook:** `Desafio_1.ipynb`

**Trabajo realizado**

- Limpieza y vectorización de documentos con **TF-IDF**.
- Exploración de **similitud entre documentos**: selección de 5 documentos aleatorios y análisis de sus vecinos más cercanos según distancia coseno.
- Implementación de un clasificador **1-NN / prototípico** basado en TF-IDF como línea base.
- Entrenamiento y ajuste de **Naïve Bayes Multinomial** y **ComplementNB**.
- Obtención de **embeddings de palabras basados en TF-IDF** (trasponiendo la matriz) y estudio de vecinos semánticos.

**Habilidades practicadas**

- Pipeline clásico de NLP.
- Similaridad semántica y recuperación de documentos.
- Evaluación con **macro F1** y análisis cualitativo de resultados.

---

## Desafío 2 – Word2Vec entrenado sobre letras de Nirvana

<!-- ![Desafío 2](images/desafio2.png) -->

**Objetivo.** Entrenar desde cero embeddings tipo **Word2Vec (Skip-gram)** usando letras de canciones de **Nirvana** y analizar la semántica aprendida.

**Notebook:** `Desafio_2.ipynb`

**Trabajo realizado**

- Construcción del corpus a partir de `songs_dataset/nirvana.txt`.
- Tokenización de cada línea en secuencias.
- Entrenamiento de **Word2Vec** con `gensim` y callback personalizado para registrar la pérdida por época.
- Exploración de palabras similares y disímiles para términos clave.
- Reducción de dimensionalidad con **t-SNE** para visualizar clusters semánticos.

**Habilidades practicadas**

- Entrenamiento de **embeddings distribuidos**.
- Uso de métricas de similitud coseno.
- Visualización y análisis cualitativo del espacio de embeddings.

---

## Desafío 3 – Modelo de Lenguaje a Nivel Carácter (SimpleRNN)

<!-- ![Desafío 3](images/desafio3.png) -->

**Objetivo.** Implementar un **modelo de lenguaje carácter-a-carácter** entrenado sobre *Así habló Zaratustra* (Nietzsche), evaluado mediante **perplejidad** y generación de texto.

**Notebook:** `Desafio_3.ipynb`

**Trabajo realizado**

- Descarga y parsing del texto usando **BeautifulSoup + lxml**, normalizando minúsculas y removiendo artefactos como `\xad`.
- Construcción del vocabulario de caracteres y dataset supervisado many-to-many con secuencias desplazadas.
- Modelo Keras:
  - `TimeDistributed(CategoryEncoding)` para codificación one-hot.
  - **SimpleRNN** con 200 unidades, `dropout` y `recurrent_dropout`.
  - `Dense + softmax` sobre el vocabulario.
- Implementación de un callback personalizado para calcular **perplejidad** en validación, guardar el mejor modelo y cortar el entrenamiento si deja de mejorar.
- Generación de texto usando *greedy decoding* y *beam search* con distintos *temperatures*.

**Habilidades practicadas**

- Preprocesamiento avanzado para modelos secuenciales.
- Implementación de métricas personalizadas (perplejidad).
- Construcción y evaluación de **modelos generativos**.

---

## Desafío 4 – Traductor Seq2Seq LSTM (Inglés → Español)

<!-- ![Desafío 4](images/desafio4.png) -->

**Objetivo.** Construir un traductor **encoder–decoder LSTM** para traducir oraciones cortas del inglés al español utilizando el corpus paralelo **Tatoeba** y embeddings preentrenados **GloVe**.

**Notebook:** `Desafio_4.ipynb`

**Trabajo realizado**

- Descarga del dataset `spa-eng/spa.txt`, muestreo y preparación de ~6600 pares oración-oración.
- Generación de tres listas alineadas:
  - `input_sentences` (inglés),
  - `output_sentences` (español + `<eos>`),
  - `output_sentences_inputs` (`<sos>` + español).
- Tokenización independiente por idioma y **padding** apropiado para encoder y decoder.
- Carga de embeddings **GloVe** en una matriz fija para el encoder.
- Construcción del modelo seq2seq:
  - **Encoder:** Embedding → LSTM → estados finales.
  - **Decoder:** Embedding → LSTM inicializado con estados del encoder → Dense + softmax.
- Entrenamiento con `Adam` + `categorical_crossentropy`, evaluación cuantitativa y pruebas cualitativas de traducción.

**Habilidades practicadas**

- Pipeline completo de **Traducción Automática Neuronal**.
- Manejo de SOS/EOS, secuencias de distinta longitud y targets one-hot.
- Uso de embeddings preentrenados dentro de modelos Keras.




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



Ejemplo de instalación:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
