"""
LECCION 1.2 - RED NEURONAL DE 2 CAPAS
=====================================

Objetivo: ver por que una sola neurona no alcanza para ciertos problemas
y como una capa oculta lo resuelve.

En la leccion 1.1 el perceptron separo dos nubes con una linea recta.
Aqui usamos el problema XOR: 4 nubes de puntos donde las nubes de una
misma diagonal tienen la misma clase. Ninguna recta puede separarlas.

La red de esta leccion tiene 3 capas de neuronas (se cuentan 2 capas
"con pesos", por eso se llama red de 2 capas):
  1. Entrada: los 2 features de cada punto (x1, x2)
  2. Capa oculta: 4 neuronas sigmoid -> cada una dibuja su propia recta
  3. Capa de salida: 1 neurona sigmoid -> combina las 4 rectas en una
     frontera curva
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: GENERAR DATOS XOR
# ============================================================
# XOR ("o exclusivo") vale 1 cuando exactamente una de las dos entradas
# es 1:  (0,0)->0  (1,1)->0  (0,1)->1  (1,0)->1
# En vez de usar solo esos 4 puntos, generamos una nube alrededor de cada
# uno (mismo mecanismo que en la 1.1: randn * dispersion + centro).
# Las nubes de la diagonal (0,0)-(1,1) son clase 0 y las de la otra
# diagonal (0,1)-(1,0) son clase 1.

def generar_datos_xor(n_por_nube=50, dispersion=0.15):
    """Genera 4 nubes de puntos con el patron XOR (no separable con una recta)."""
    centros = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    etiquetas = [0, 0, 1, 1]

    nubes = [np.random.randn(n_por_nube, 2) * dispersion + c for c in centros]
    X = np.vstack(nubes)                                                # (200, 2)
    y = np.hstack([np.full(n_por_nube, e, dtype=float) for e in etiquetas])  # (200,)

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generar_datos_circulos(n_por_clase=100):
    """EXPERIMENTO: un circulo (clase 0) rodeado por un anillo (clase 1)."""
    angulos = np.random.uniform(0, 2 * np.pi, 2 * n_por_clase)
    radios = np.hstack([np.random.randn(n_por_clase) * 0.15 + 0.5,     # circulo interior
                        np.random.randn(n_por_clase) * 0.15 + 1.5])    # anillo exterior
    X = np.column_stack([radios * np.cos(angulos), radios * np.sin(angulos)])
    y = np.hstack([np.zeros(n_por_clase), np.ones(n_por_clase)])
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


X, y = generar_datos_xor()
print(f"Datos generados: X.shape={X.shape}, y.shape={y.shape}")
print(f"Ejemplo de fila: X[0]={X[0]}, y[0]={y[0]}")


# ============================================================
# PARTE 2: EL PERCEPTRON DE LA LECCION 1.1 NO ALCANZA
# ============================================================
# Es exactamente la misma neurona de la 1.1: z = X @ w + b, sigmoid(z).
# Su frontera de decision es donde z = 0, y eso siempre es una RECTA.
# Con XOR, cualquier recta deja al menos una nube del lado equivocado.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def entrenar_perceptron(X, y, tasa_aprendizaje=0.5, epocas=3000):
    w = np.random.randn(X.shape[1]) * 0.1
    b = 0.0
    historial_error = []
    for epoca in range(epocas):
        y_pred = sigmoid(X @ w + b)
        error = y - y_pred
        derivada = y_pred * (1 - y_pred)
        w += tasa_aprendizaje * X.T @ (error * derivada) / len(X)
        b += tasa_aprendizaje * np.mean(error * derivada)
        historial_error.append(np.mean(error ** 2))
    return w, b, historial_error


print("\n--- Perceptron de la leccion 1.1 sobre XOR ---")
w_p, b_p, historial_p = entrenar_perceptron(X, y)
accuracy_p = np.mean((sigmoid(X @ w_p + b_p) >= 0.5) == y)
print(f"Accuracy del perceptron: {accuracy_p:.2%}  (adivinar al azar daria ~50%)")


# ============================================================
# PARTE 3: LA RED DE 2 CAPAS (forward pass)
# ============================================================
# Cada neurona oculta es un perceptron como el de la 1.1. Con 4 neuronas
# ocultas, sus pesos se guardan juntos en una matriz W1 de (2, 4):
#   - fila    = una entrada (x1 o x2)
#   - columna = una neurona oculta
#
#   z1 = X @ W1 + b1       (n,2) @ (2,4) + (4,)  -> (n,4)
#   a1 = sigmoid(z1)       4 "features nuevos" por punto, entre 0 y 1
#   z2 = a1 @ W2 + b2      (n,4) @ (4,1) + (1,)  -> (n,1)
#   y_pred = sigmoid(z2)   probabilidad de clase 1
#
# La neurona de salida no ve x1 y x2: ve las 4 salidas a1 de la capa
# oculta. Por eso puede separar con una recta en ESE espacio lo que no se
# podia separar con una recta en el espacio original.

def inicializar_red(n_entrada=2, n_oculta=4, n_salida=1):
    W1 = np.random.randn(n_entrada, n_oculta)    # (2, 4) pesos aleatorios (no todos iguales)
    b1 = np.zeros(n_oculta)                      # (4,)
    W2 = np.random.randn(n_oculta, n_salida)     # (4, 1)
    b2 = np.zeros(n_salida)                      # (1,)
    return W1, b1, W2, b2


def forward_red(X, W1, b1, W2, b2):
    """Devuelve la salida de la capa oculta (a1) y la prediccion final."""
    a1 = sigmoid(X @ W1 + b1)          # (n, 4)
    y_pred = sigmoid(a1 @ W2 + b2)     # (n, 1)
    return a1, y_pred


# ============================================================
# PARTE 4: ENTRENAMIENTO
# ============================================================
# El loop es el mismo de la 1.1 (predecir -> error -> gradiente ->
# actualizar), con una novedad: tambien hay que corregir W1, y el error
# solo se conoce en la salida. La solucion es repartir el error hacia
# atras: cada neurona oculta recibe una parte del error segun el peso que
# la conecta con la salida (delta2 @ W2.T). Eso es backpropagation, y la
# leccion 1.3 lo deriva paso a paso. Aqui alcanza con ver que cada capa
# usa la misma formula "error * derivada" que ya conoces.
#
# OJO con las formas: y_pred sale (n,1) pero y es (n,). Restarlos directo
# daria (n,n) por broadcasting. Por eso convertimos y en una columna (n,1).

def entrenar_red(X, y, n_oculta=4, tasa_aprendizaje=2.0, epocas=3000):
    W1, b1, W2, b2 = inicializar_red(X.shape[1], n_oculta, 1)
    Y = y.reshape(-1, 1)                     # (n,) -> (n,1)
    historial_error = []

    for epoca in range(epocas):
        a1, y_pred = forward_red(X, W1, b1, W2, b2)
        error = Y - y_pred                               # (n,1)

        # Capa de salida: igual que el perceptron de la 1.1
        delta2 = error * y_pred * (1 - y_pred)           # (n,1)
        # Capa oculta: el error de la salida se reparte con los pesos W2
        delta1 = (delta2 @ W2.T) * a1 * (1 - a1)         # (n,1) @ (1,4) -> (n,4)

        W2 += tasa_aprendizaje * a1.T @ delta2 / len(X)  # (4,n) @ (n,1) -> (4,1)
        b2 += tasa_aprendizaje * np.mean(delta2, axis=0)
        W1 += tasa_aprendizaje * X.T @ delta1 / len(X)   # (2,n) @ (n,4) -> (2,4)
        b1 += tasa_aprendizaje * np.mean(delta1, axis=0)

        error_cuadratico_medio = np.mean(error ** 2)
        historial_error.append(error_cuadratico_medio)

        if epoca % 500 == 0 or epoca == epocas - 1:
            accuracy = np.mean((y_pred >= 0.5) == Y)
            print(f"Epoca {epoca:4d} | error={error_cuadratico_medio:.4f} | accuracy={accuracy:.2%}")

    return W1, b1, W2, b2, historial_error


print("\n--- Entrenando la red de 2 capas sobre XOR ---")
W1, b1, W2, b2, historial_red = entrenar_red(X, y)


# ============================================================
# PARTE 5: EVALUACION
# ============================================================

_, y_pred_final = forward_red(X, W1, b1, W2, b2)
accuracy_red = np.mean((y_pred_final.ravel() >= 0.5) == y)
print(f"\nAccuracy del perceptron:    {accuracy_p:.2%}")
print(f"Accuracy de la red 2 capas: {accuracy_red:.2%}")
print(f"\nW1 (2,4) =\n{np.round(W1, 2)}")
print(f"b1 (4,)  = {np.round(b1, 2)}")
print(f"W2 (4,1) = {np.round(W2.ravel(), 2)}  b2 = {np.round(b2, 2)}")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================
# (1) el perceptron con su unica recta, (2) la red con su frontera curva
# y las 4 rectas de sus neuronas ocultas, (3) las curvas de aprendizaje.

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4


def dibujar_puntos(ax):
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="#3B82F6", label="Clase 0", edgecolor="white")
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="#EF4444", label="Clase 1", edgecolor="white")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# --- Grafico 1: perceptron (una sola recta) ---
dibujar_puntos(ax1)
x1_vals = np.linspace(x_min, x_max, 100)
ax1.plot(x1_vals, -(w_p[0] * x1_vals + b_p) / w_p[1], color="black", linestyle="--", label="Frontera")
ax1.set_title(f"Perceptron (1.1): una recta -> {accuracy_p:.0%}")
ax1.legend(loc="upper right")

# --- Grafico 2: red de 2 capas (frontera curva) ---
# Evaluamos la red en una grilla de 300x300 puntos y pintamos cada zona
# segun la clase que predice. contour(..., levels=[0.5]) dibuja la frontera.
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grilla = np.column_stack([xx.ravel(), yy.ravel()])            # (90000, 2)
_, prob_grilla = forward_red(grilla, W1, b1, W2, b2)
prob_grilla = prob_grilla.reshape(xx.shape)
ax2.contourf(xx, yy, prob_grilla, levels=[0, 0.5, 1], colors=["#DBEAFE", "#FEE2E2"], alpha=0.8)
ax2.contour(xx, yy, prob_grilla, levels=[0.5], colors="black", linestyles="--")
# Rectas de cada neurona oculta: donde su z1 = 0 -> x2 = -(W1[0,j]*x1 + b1[j]) / W1[1,j]
for j in range(W1.shape[1]):
    ax2.plot(x1_vals, -(W1[0, j] * x1_vals + b1[j]) / W1[1, j], color="gray", linewidth=0.8, alpha=0.7)
dibujar_puntos(ax2)
ax2.set_title(f"Red 2 capas: frontera curva -> {accuracy_red:.0%}")

# --- Grafico 3: curvas de aprendizaje ---
ax3.plot(historial_p, color="#F59E0B", label="Perceptron")
ax3.plot(historial_red, color="#10B981", label="Red 2 capas")
ax3.set_title("Curva de aprendizaje")
ax3.set_xlabel("Epoca")
ax3.set_ylabel("Error cuadratico medio")
ax3.legend()

plt.tight_layout()
plt.savefig("03_neural_network_2layer.png", dpi=120)
print("\nGrafico guardado en: 03_neural_network_2layer.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Una sola neurona solo dibuja fronteras RECTAS: no puede resolver XOR
2. Una capa oculta = varias neuronas en paralelo; cada una dibuja su recta
   y entrega un "feature nuevo" (a1) entre 0 y 1
3. La neurona de salida combina esos features nuevos: asi aparece una
   frontera curva en el espacio original
4. Los pesos de una capa se guardan como matriz: W1 (entradas, neuronas)
5. Entrenar la capa oculta requiere repartir el error hacia atras
   (delta2 @ W2.T) -> la leccion 1.3 lo explica en detalle

EXPERIMENTOS SUGERIDOS:
- Cambia n_oculta a 1, 2 y 8 en entrenar_red(): cual es el minimo que
  resuelve XOR?
- Inicializa W1 y W2 con np.zeros(...): por que la red no aprende?
- Reemplaza generar_datos_xor() por generar_datos_circulos(): la misma
  red puede con un circulo dentro de un anillo?
""")
