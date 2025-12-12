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


```bash
Input: My mother says hi
Representacion en vector de tokens de ids [16, 236, 346]
Padding del vector: [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16
  236 346]]
Input: My mother says hi
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
Response: ella se ha ido

______________________________________________________

Input: Buckle your seatbelt, Dorothy, because Kansas is going bye bye.
Representacion en vector de tokens de ids [29, 215, 7, 78]
Padding del vector: [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  29 215
    7  78]]
Input: Buckle your seatbelt, Dorothy, because Kansas is going bye bye.
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
Response: tu esposa es muy joven

______________________________________________________

Input: Follow the white rabbit
Representacion en vector de tokens de ids [513, 1, 742]
Padding del vector: [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 513
    1 742]]
Input: Follow the white rabbit
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
Response: ve a tom

______________________________________________________

Input: Why oh why didn't I take the blue pill?
Representacion en vector de tokens de ids [74, 2427, 74, 54, 3, 102, 1, 1156]
Padding del vector: [[   0    0    0    0    0    0    0    0    0    0    0    0   74 2427
    74   54    3  102    1 1156]]
Input: Why oh why didn't I take the blue pill?
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
Response: qué qué hora es el libro

______________________________________________________

Input: There is no spoon.
Representacion en vector de tokens de ids [51, 7, 61, 3280]
Padding del vector: [[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0   51    7   61 3280]]
Input: There is no spoon.
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
Response: tom es un buen amigo

______________________________________________________

Input: Human beings are a disease, a cancer of this planet.
Representacion en vector de tokens de ids [1840, 22, 6, 1128, 6, 1565, 10, 17, 2166]
Padding del vector: [[   0    0    0    0    0    0    0    0    0    0    0 1840   22    6
  1128    6 1565   10   17 2166]]
Input: Human beings are a disease, a cancer of this planet.
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
Response: él y no le dio un poco de que él se ha pasado

______________________________________________________

Input: Never send a human to do a machine's job.
Representacion en vector de tokens de ids [119, 1113, 6, 1840, 2, 12, 6, 200]
Padding del vector: [[   0    0    0    0    0    0    0    0    0    0    0    0  119 1113
     6 1840    2   12    6  200]]
Input: Never send a human to do a machine's job.
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
Response: él se puso a mi auto de la casa de mary

______________________________________________________

Input: There's a difference between knowing the path and walking the path.
Representacion en vector de tokens de ids [194, 6, 1081, 1065, 3558, 1, 33, 648, 1]
Padding del vector: [[   0    0    0    0    0    0    0    0    0    0    0  194    6 1081
  1065 3558    1   33  648    1]]
Input: There's a difference between knowing the path and walking the path.
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
Response: tom y mary y mary se ha sido pasado a mary

______________________________________________________

Input: Would you still have broken it if I hadn't said anything?
Representacion en vector de tokens de ids [76, 4, 154, 18, 911, 14, 64, 3, 783, 122, 114]
Padding del vector: [[  0   0   0   0   0   0   0   0   0  76   4 154  18 911  14  64   3 783
  122 114]]
Input: Would you still have broken it if I hadn't said anything?
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
Response: si qué te lo que te gusta lo que me gusta nada

______________________________________________________