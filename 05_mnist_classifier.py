"""
LECCION 1.4 - CASO REAL: MNIST
==============================

Objetivo: usar la red de las lecciones 1.2 y 1.3 en un problema real:
reconocer digitos escritos a mano (0 al 9).

MNIST tiene 70 000 imagenes de 28x28 pixeles en escala de grises:
  - 60 000 para entrenar (train)
  - 10 000 para evaluar (test), que la red NUNCA ve durante el entrenamiento

Lo nuevo respecto a XOR:
  1. Los datos: cada imagen se aplana a un vector de 784 numeros
  2. La salida: 10 neuronas (una por digito) con softmax, y la perdida
     cross-entropy en vez del error cuadratico
  3. El entrenamiento: mini-batches de 64 imagenes en vez de usar las
     60 000 en cada actualizacion
  4. La evaluacion: accuracy sobre datos que la red no vio (test)

La primera vez descarga MNIST (~11 MB) a la carpeta data/.
"""

import gzip
import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: DESCARGAR Y LEER MNIST
# ============================================================
# Los archivos vienen comprimidos (.gz) en formato IDX: unos bytes de
# cabecera y despues los pixeles uno tras otro, cada uno un byte (0-255).
#   - imagenes: cabecera de 16 bytes, despues 28*28 bytes por imagen
#   - etiquetas: cabecera de 8 bytes, despues 1 byte por imagen

URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
ARCHIVOS = {
    "X_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "X_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def cargar_mnist(carpeta="data"):
    os.makedirs(carpeta, exist_ok=True)
    datos = {}
    for nombre, archivo in ARCHIVOS.items():
        ruta = os.path.join(carpeta, archivo)
        if not os.path.exists(ruta):
            print(f"Descargando {archivo}...")
            urllib.request.urlretrieve(URL_BASE + archivo, ruta)
        with gzip.open(ruta, "rb") as f:
            contenido = f.read()
        if nombre.startswith("X"):
            pixeles = np.frombuffer(contenido, dtype=np.uint8, offset=16)
            datos[nombre] = pixeles.reshape(-1, 28, 28)       # (n, 28, 28)
        else:
            datos[nombre] = np.frombuffer(contenido, dtype=np.uint8, offset=8)  # (n,)
    return datos["X_train"], datos["y_train"], datos["X_test"], datos["y_test"]


imagenes_train, y_train, imagenes_test, y_test = cargar_mnist()
print(f"Train: {imagenes_train.shape} imagenes, {y_train.shape} etiquetas")
print(f"Test:  {imagenes_test.shape} imagenes, {y_test.shape} etiquetas")


# ============================================================
# PARTE 2: PREPARAR LOS DATOS
# ============================================================
# (a) Aplanar: la red recibe vectores, no matrices. Cada imagen de 28x28
#     se convierte en una fila de 784 numeros (las filas de pixeles una
#     detras de otra).
# (b) Normalizar: los pixeles van de 0 a 255. Dividiendo por 255 quedan
#     entre 0 y 1, del mismo orden que las entradas de XOR.
# (c) One-hot: la etiqueta 3 se convierte en [0,0,0,1,0,0,0,0,0,0], un
#     vector con un 1 en la posicion de la clase correcta. Asi se puede
#     comparar con las 10 salidas de la red.

X_train = imagenes_train.reshape(-1, 784) / 255.0      # (60000, 784)
X_test = imagenes_test.reshape(-1, 784) / 255.0        # (10000, 784)


def one_hot(etiquetas, n_clases=10):
    Y = np.zeros((len(etiquetas), n_clases))
    Y[np.arange(len(etiquetas)), etiquetas] = 1.0
    return Y


Y_train = one_hot(y_train)                             # (60000, 10)
print(f"\nX_train: {X_train.shape}, valores entre {X_train.min()} y {X_train.max()}")
print(f"Etiqueta y_train[0]={y_train[0]} -> one-hot {Y_train[0]}")


# ============================================================
# PARTE 3: LA RED (784 -> 64 -> 10)
# ============================================================
# Misma estructura que la 1.3, con dos cambios en la salida:
#   - 10 neuronas, una por digito
#   - softmax en vez de sigmoid: convierte los 10 numeros z2 en 10
#     probabilidades que suman 1
#
# Inicializacion: con 784 entradas, randn sin achicar daria z1 enormes
# (suma de 784 terminos) y la sigmoid se saturaria desde el inicio.
# Multiplicar por sqrt(1/n_entradas) mantiene z1 en un rango razonable.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)       # restar el maximo evita overflow en exp
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)    # cada fila suma 1


def inicializar_red(n_entrada=784, n_oculta=64, n_salida=10):
    return {
        "W1": np.random.randn(n_entrada, n_oculta) * np.sqrt(1.0 / n_entrada),
        "b1": np.zeros(n_oculta),
        "W2": np.random.randn(n_oculta, n_salida) * np.sqrt(1.0 / n_oculta),
        "b2": np.zeros(n_salida),
    }


def forward(X, p):
    a1 = sigmoid(X @ p["W1"] + p["b1"])        # (n, 64)
    probs = softmax(a1 @ p["W2"] + p["b2"])    # (n, 10)
    return probs, {"X": X, "a1": a1, "probs": probs}


def perdida_cross_entropy(probs, Y):
    """Promedio de -log(probabilidad que la red le dio a la clase correcta)."""
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def backward(Y, p, cache):
    n = len(Y)
    X, a1, probs = cache["X"], cache["a1"], cache["probs"]
    # Con softmax + cross-entropy, la derivada respecto a z2 queda muy simple:
    dZ2 = (probs - Y) / n                      # (n, 10)
    dW2 = a1.T @ dZ2                           # (64, 10)
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ p["W2"].T                      # (n, 64)  el mismo patron de la 1.3
    dZ1 = dA1 * a1 * (1 - a1)
    dW1 = X.T @ dZ1                            # (784, 64)
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(X, y, p):
    probs, _ = forward(X, p)
    return np.mean(np.argmax(probs, axis=1) == y)   # la clase predicha es la de mayor probabilidad


# ============================================================
# PARTE 4: ENTRENAMIENTO CON MINI-BATCHES
# ============================================================
# En vez de calcular el gradiente con las 60 000 imagenes (full-batch,
# como en las lecciones anteriores), se usa un grupo de 64 por vez.
# Una epoca = recorrer las 60 000 en 938 grupos = 938 actualizaciones.
# Al inicio de cada epoca se mezcla el orden: ahora SI importa, porque
# cada grupo debe ser una muestra representativa de todos los digitos.

def entrenar(X, Y, y, X_test, y_test, tasa_aprendizaje=0.5, epocas=10, tamano_batch=64):
    params = inicializar_red()
    historial = {"perdida": [], "acc_train": [], "acc_test": []}
    n = len(X)
    for epoca in range(epocas):
        orden = np.random.permutation(n)                 # mezclar en cada epoca
        for inicio in range(0, n, tamano_batch):
            idx = orden[inicio:inicio + tamano_batch]    # los indices de este mini-batch
            probs, cache = forward(X[idx], params)
            grads = backward(Y[idx], params, cache)
            for nombre in params:
                params[nombre] -= tasa_aprendizaje * grads[nombre]

        probs_train, _ = forward(X, params)
        historial["perdida"].append(perdida_cross_entropy(probs_train, Y))
        historial["acc_train"].append(np.mean(np.argmax(probs_train, axis=1) == y))
        historial["acc_test"].append(accuracy(X_test, y_test, params))
        print(f"Epoca {epoca + 1:2d} | perdida={historial['perdida'][-1]:.4f} | "
              f"train={historial['acc_train'][-1]:.2%} | test={historial['acc_test'][-1]:.2%}")
    return params, historial


p_inicial = inicializar_red()
print(f"\nAntes de entrenar: accuracy en test = {accuracy(X_test, y_test, p_inicial):.2%} (al azar ~10%)")

print("\n--- Entrenando (10 epocas x 938 mini-batches) ---")
params, historial = entrenar(X_train, Y_train, y_train, X_test, y_test)


# ============================================================
# PARTE 5: EVALUACION SOBRE TEST
# ============================================================
# La matriz de confusion cuenta, para cada digito real (fila), que
# predijo la red (columna). La diagonal son los aciertos.

probs_test, _ = forward(X_test, params)
pred_test = np.argmax(probs_test, axis=1)
print(f"\nAccuracy final en test: {np.mean(pred_test == y_test):.2%}")

confusion = np.zeros((10, 10), dtype=int)
for real, pred in zip(y_test, pred_test):
    confusion[real, pred] += 1
print("\nAccuracy por digito:")
for d in range(10):
    print(f"  {d}: {confusion[d, d] / confusion[d].sum():.2%}")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================
# (1) primeras 10 imagenes de test con la prediccion,
# (2) accuracy train vs test por epoca, (3) matriz de confusion

fig = plt.figure(figsize=(16, 5.5))
grilla = fig.add_gridspec(1, 3)                 # 3 columnas iguales
grilla_imagenes = grilla[0].subgridspec(2, 5)   # la primera, dividida en 2x5

for i in range(10):
    ax = fig.add_subplot(grilla_imagenes[i // 5, i % 5])
    ax.imshow(imagenes_test[i], cmap="gray_r")
    color = "#10B981" if pred_test[i] == y_test[i] else "#EF4444"
    ax.set_title(f"pred {pred_test[i]}", fontsize=9, color=color)
    ax.axis("off")

ax2 = fig.add_subplot(grilla[1])
epocas_x = range(1, len(historial["acc_train"]) + 1)
ax2.plot(epocas_x, historial["acc_train"], color="#3B82F6", marker="o", label="Train")
ax2.plot(epocas_x, historial["acc_test"], color="#F59E0B", marker="o", label="Test")
ax2.set_title("Accuracy por epoca")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("Accuracy")
ax2.legend()

ax3 = fig.add_subplot(grilla[2])
ax3.imshow(np.log1p(confusion), cmap="Blues")
ax3.set_xticks(range(10))
ax3.set_yticks(range(10))
ax3.set_xlabel("Prediccion")
ax3.set_ylabel("Digito real")
ax3.set_title("Matriz de confusion (test)")
for r in range(10):
    for c in range(10):
        if confusion[r, c] > 0:
            ax3.text(c, r, confusion[r, c], ha="center", va="center", fontsize=7,
                     color="white" if r == c else "black")

plt.tight_layout()
plt.savefig("05_mnist_classifier.png", dpi=120)
print("\nGrafico guardado en: 05_mnist_classifier.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Una imagen es una matriz de numeros; aplanada, es un vector de features
2. Normalizar las entradas (0-1) y escalar los pesos iniciales evita que
   la sigmoid se sature desde el principio
3. Para varias clases: one-hot en las etiquetas, softmax en la salida y
   cross-entropy como perdida. Su gradiente es simplemente probs - Y
4. Mini-batches: muchas actualizaciones por epoca con gradientes
   aproximados -> se aprende mucho mas rapido que con full-batch
5. El accuracy que importa es el de test: datos que la red nunca vio

EXPERIMENTOS SUGERIDOS:
- Quita el "/ 255.0" de la Parte 2: que pasa con la perdida?
- Cambia tamano_batch a 1, a 600 y a 60000 (full-batch): compara la
  velocidad y el accuracy despues de 10 epocas
- Cambia n_oculta a 16 y a 256: cuanto mejora el accuracy en test?
""")
