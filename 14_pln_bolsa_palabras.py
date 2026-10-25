"""
LECCION 3.2 - BOLSA DE PALABRAS Y TF-IDF
========================================

Objetivo: representar un documento entero como un vector de numeros, y
usar ese vector para clasificar.

En la leccion 3.1 cada palabra paso a ser un indice. Pero un clasificador
como la neurona de la 1.1 necesita un vector de tamano FIJO por ejemplo,
y las resenas tienen largos distintos. La "bolsa de palabras" resuelve
eso: un vector con una posicion por palabra del vocabulario, que cuenta
cuantas veces aparece cada una. Se pierde el orden; se conserva el tema.

Tarea: decidir si una resena es positiva (1) o negativa (0), con las 80
resenas de corpus_es.py.
"""

from collections import Counter
import re

import matplotlib.pyplot as plt
import numpy as np

from corpus_es import RESENAS

np.random.seed(42)


# ============================================================
# PARTE 1: TEXTO, TOKENS Y DIVISION TRAIN / TEST
# ============================================================

def tokenizar(texto):
    """La misma tokenizacion de la leccion 3.1."""
    return re.findall(r"[a-záéíóúüñ]+", texto.lower())


textos = [t for t, _ in RESENAS]
y = np.array([e for _, e in RESENAS], dtype=float)

# 60 para entrenar y 20 para evaluar, mezcladas (como en la 1.1)
indices = np.random.permutation(len(textos))
idx_train, idx_test = indices[:60], indices[60:]
print(f"Train: {len(idx_train)} resenas ({y[idx_train].mean():.0%} positivas) | "
      f"Test: {len(idx_test)} ({y[idx_test].mean():.0%} positivas)")


# ============================================================
# PARTE 2: LA BOLSA DE PALABRAS
# ============================================================
# El vocabulario se arma SOLO con las resenas de train: las palabras que
# aparecen unicamente en test se ignoran (como pasaria con texto nuevo).
# Cada resena -> vector de largo V; posicion j = cuantas veces aparece la
# palabra j.

vocab = {}
for i in idx_train:
    for t in tokenizar(textos[i]):
        if t not in vocab:
            vocab[t] = len(vocab)
V = len(vocab)


def bolsa_de_palabras(texto):
    vector = np.zeros(V)
    for t in tokenizar(texto):
        if t in vocab:
            vector[vocab[t]] += 1
    return vector


X_cuentas = np.array([bolsa_de_palabras(t) for t in textos])     # (80, V)
print(f"\nVocabulario de train: {V} palabras -> X tiene shape {X_cuentas.shape}")
ejemplo = textos[0]
v = X_cuentas[0]
print(f"Resena 0: '{ejemplo}'")
print(f"  posiciones no nulas: {int((v > 0).sum())} de {V} ->",
      {p: int(v[j]) for p, j in vocab.items() if v[j] > 0})


# ============================================================
# PARTE 3: TF-IDF
# ============================================================
# Problema de las cuentas: "y", "el", "muy" aparecen en todas las resenas
# y pesan tanto como "deliciosa". TF-IDF multiplica cada cuenta (TF, term
# frequency) por un factor que premia las palabras RARAS:
#   IDF(palabra) = log(N / df) + 1
#   N  = cantidad de documentos de train
#   df = en cuantos documentos de train aparece la palabra
# Despues se normaliza cada fila a largo 1, para que las resenas largas
# no tengan vectores mas grandes solo por ser largas.

N = len(idx_train)
df = (X_cuentas[idx_train] > 0).sum(axis=0)          # (V,) documentos que contienen cada palabra
idf = np.log(N / df) + 1


def tfidf(X):
    W = X * idf                                       # broadcasting: (n, V) * (V,)
    normas = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.where(normas == 0, 1, normas)


X_tfidf = tfidf(X_cuentas)
for p in ["y", "el", "muy", "deliciosa", "horrible", "pesimo"]:
    if p in vocab:
        j = vocab[p]
        print(f"  '{p}': aparece en {int(df[j])} de {N} resenas -> IDF = {idf[j]:.2f}")


# ============================================================
# PARTE 4: CLASIFICAR CON LA NEURONA DE LA LECCION 1.1
# ============================================================
# Una sola neurona sigmoid: z = X @ w + b. Ahora cada "feature" es una
# palabra, y su peso dice cuanto empuja hacia "positiva".
# Se usa cross-entropy (leccion 1.4), cuyo gradiente es (y_pred - y).

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def entrenar(X, y, tasa_aprendizaje=1.0, epocas=500):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epocas):
        y_pred = sigmoid(X @ w + b)
        error = y_pred - y
        w -= tasa_aprendizaje * X.T @ error / len(X)
        b -= tasa_aprendizaje * error.mean()
    return w, b


def accuracy(X, y, w, b):
    return np.mean((sigmoid(X @ w + b) >= 0.5) == y)


tokens_test = [t for i in idx_test for t in tokenizar(textos[i])]
desconocidas = sum(t not in vocab for t in tokens_test)
print(f"\nPalabras de test que no estan en el vocabulario de train: {desconocidas} de {len(tokens_test)}")

print("\n--- Clasificador (una neurona) ---")
resultados = {}
for nombre, X in [("Cuentas (bolsa de palabras)", X_cuentas), ("TF-IDF", X_tfidf)]:
    w, b = entrenar(X[idx_train], y[idx_train])
    resultados[nombre] = (w, b)
    print(f"{nombre:28s} | train={accuracy(X[idx_train], y[idx_train], w, b):.0%} | "
          f"test={accuracy(X[idx_test], y[idx_test], w, b):.0%}")


# ============================================================
# PARTE 5: QUE APRENDIO (los pesos son interpretables)
# ============================================================

w, b = resultados["TF-IDF"]
palabras = np.array(list(vocab))
orden = np.argsort(w)
print("\nPalabras mas NEGATIVAS:", ", ".join(f"{palabras[j]} ({w[j]:.2f})" for j in orden[:8]))
print("Palabras mas POSITIVAS:", ", ".join(f"{palabras[j]} ({w[j]:.2f})" for j in orden[::-1][:8]))


# ============================================================
# PARTE 6: FRASES NUEVAS Y EL LIMITE DE LA BOLSA
# ============================================================
# La bolsa no tiene orden: "no es buena, es mala" y "no es mala, es
# buena" son el MISMO vector.

nuevas = [
    "La atencion fue excelente y la comida deliciosa",
    "Una experiencia terrible, todo llego roto",
    "No es buena, es mala",
    "No es mala, es buena",
    "Nada que ver con la descripcion, pero me encanto",
]
print("\n--- Frases nuevas ---")
for frase in nuevas:
    p = sigmoid(tfidf(bolsa_de_palabras(frase)[None, :]) @ w + b)[0]
    print(f"P(positiva) = {p:.2f} | {frase}")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

seleccion = np.concatenate([orden[:10], orden[-10:]])
colores = ["#EF4444" if w[j] < 0 else "#10B981" for j in seleccion]
ax1.barh(palabras[seleccion], w[seleccion], color=colores)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_title("Pesos aprendidos (TF-IDF): las 10 palabras mas negativas y mas positivas")
ax1.set_xlabel("Peso w (positivo = empuja hacia 'resena positiva')")

ax2.imshow(X_tfidf[np.argsort(y)][:, np.argsort(-df)[:40]], aspect="auto", cmap="Blues")
ax2.axhline(39.5, color="#EF4444", linewidth=1.5)
ax2.set_title("Matriz TF-IDF: 80 resenas x 40 palabras mas comunes")
ax2.set_xlabel("Palabras (de mas a menos documentos)")
ax2.set_ylabel("Resenas (arriba negativas, abajo positivas)")

plt.tight_layout()
plt.savefig("14_pln_bolsa_palabras.png", dpi=120)
print("\nGrafico guardado en: 14_pln_bolsa_palabras.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Bolsa de palabras: un documento -> un vector de cuentas, de largo fijo
   (el tamano del vocabulario). Se pierde el orden
2. TF-IDF baja el peso de las palabras que estan en todos los documentos
   y sube el de las que distinguen
3. Con vectores de largo fijo sirve cualquier clasificador del modulo 1:
   aqui, una sola neurona (regresion logistica)
4. Los pesos de un modelo lineal se pueden leer: dicen que palabras
   empujan hacia cada clase
5. Sin orden, "no es buena, es mala" = "no es mala, es buena"

EXPERIMENTOS SUGERIDOS:
- Entrena sin la normalizacion de largo (usa X * idf directamente)
- Agrega a la bolsa los pares de palabras seguidas ("no_es", "es_bueno"):
  se distinguen ahora las dos frases con "no"?
- Cambia el reparto a 40 train / 40 test: cuanto baja el accuracy?
""")
