"""
LECCION 1.5 - FUNCIONES DE ACTIVACION
=====================================

Objetivo: entender por que una red necesita activaciones no lineales,
comparar las 4 mas conocidas (step, sigmoid, tanh, ReLU) y ver en una red
profunda el problema que motivo el cambio de sigmoid a ReLU: los
gradientes que desaparecen.

Hasta ahora todas las capas ocultas usaron sigmoid. En esta leccion:
  1. Graficamos cada activacion y su derivada
  2. Comprobamos que sin activacion, 2 capas equivalen a 1
  3. Medimos el tamano del gradiente en cada capa de una red de 6 capas
     ocultas, con cada activacion
  4. Entrenamos esa red profunda con cada activacion y comparamos
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: LAS ACTIVACIONES Y SUS DERIVADAS
# ============================================================
# Cada activacion se define junto con su derivada, porque backprop
# necesita las dos. La derivada recibe z (la entrada de la activacion)
# y a (su salida): algunas se calculan mas facil con una, otras con otra.

def step(z):
    return (z > 0).astype(float)           # 0 o 1: la neurona "dispara" o no


def step_derivada(z, a):
    return np.zeros_like(z)                # plana en todas partes (en z = 0 no existe)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z, a):
    return a * (1 - a)                     # maximo 0.25, en z = 0


def tanh(z):
    return np.tanh(z)                      # como sigmoid, pero entre -1 y 1


def tanh_derivada(z, a):
    return 1 - a ** 2                      # maximo 1, en z = 0


def relu(z):
    return np.maximum(0, z)                # 0 si z es negativo, z si es positivo


def relu_derivada(z, a):
    return (z > 0).astype(float)           # 1 si z > 0, 0 si no


ACTIVACIONES = {
    "sigmoid": (sigmoid, sigmoid_derivada),
    "tanh": (tanh, tanh_derivada),
    "relu": (relu, relu_derivada),
}

z_demo = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("z            =", z_demo)
for nombre, (f, df) in [("step", (step, step_derivada))] + list(ACTIVACIONES.items()):
    a = f(z_demo)
    print(f"{nombre:8s} a = {np.round(a, 3)}   derivada = {np.round(df(z_demo, a), 3)}")


# ============================================================
# PARTE 2: SIN ACTIVACION, DOS CAPAS SON UNA SOLA
# ============================================================
# Si la capa oculta no aplica ninguna activacion:
#   salida = (X @ W1) @ W2 = X @ (W1 @ W2)
# W1 @ W2 es una sola matriz: la red entera es equivalente a una capa.

W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)
X_prueba = np.random.randn(5, 2)
dos_capas = (X_prueba @ W1) @ W2
una_capa = X_prueba @ (W1 @ W2)
print("\nSin activacion: (X @ W1) @ W2 =", np.round(dos_capas.ravel(), 4))
print("                X @ (W1 @ W2) =", np.round(una_capa.ravel(), 4))
print("W1 @ W2 (2,1) =", np.round((W1 @ W2).ravel(), 4), "-> una sola capa de 2 pesos")


# ============================================================
# PARTE 3: UNA RED PROFUNDA GENERICA
# ============================================================
# Datos: el circulo dentro de un anillo de la leccion 1.2.
# Red: 2 -> 16 -> 16 -> 16 -> 16 -> 16 -> 16 -> 1  (6 capas ocultas)
# Salida: sigmoid + cross-entropy binaria. Es la version de 2 clases de
# softmax + cross-entropy (leccion 1.4), y su gradiente tambien es
# simplemente (a - y) / n.
#
# Las capas se guardan en listas: W[0], W[1], ... Asi forward y backward
# son un for sobre las capas, sin importar cuantas haya.

def generar_datos_circulos(n_por_clase=100):
    angulos = np.random.uniform(0, 2 * np.pi, 2 * n_por_clase)
    radios = np.hstack([np.random.randn(n_por_clase) * 0.15 + 0.5,
                        np.random.randn(n_por_clase) * 0.15 + 1.5])
    X = np.column_stack([radios * np.cos(angulos), radios * np.sin(angulos)])
    y = np.hstack([np.zeros(n_por_clase), np.ones(n_por_clase)])
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def inicializar_red(tamanos, activacion):
    """tamanos = [2, 16, 16, ..., 1]. Escala segun la activacion:
    sqrt(1/n) para sigmoid/tanh (Xavier), sqrt(2/n) para ReLU (He)."""
    factor = 2.0 if activacion == "relu" else 1.0
    W = [np.random.randn(n_in, n_out) * np.sqrt(factor / n_in)
         for n_in, n_out in zip(tamanos[:-1], tamanos[1:])]
    b = [np.zeros(n_out) for n_out in tamanos[1:]]
    return W, b


def forward(X, W, b, activacion):
    f, _ = ACTIVACIONES[activacion]
    zs, activaciones = [], [X]                     # activaciones[0] es la entrada
    for capa in range(len(W)):
        z = activaciones[-1] @ W[capa] + b[capa]
        es_salida = capa == len(W) - 1
        a = sigmoid(z) if es_salida else f(z)      # la salida siempre es sigmoid
        zs.append(z)
        activaciones.append(a)
    return zs, activaciones


def backward(Y, W, zs, activaciones, activacion):
    _, df = ACTIVACIONES[activacion]
    n = len(Y)
    dW, db = [None] * len(W), [None] * len(W)
    dZ = (activaciones[-1] - Y) / n                # sigmoid + cross-entropy: a - y
    for capa in reversed(range(len(W))):           # de la ultima capa a la primera
        dW[capa] = activaciones[capa].T @ dZ       # entrada de la capa .T @ delta
        db[capa] = dZ.sum(axis=0)
        if capa > 0:                               # pasar el delta a la capa anterior
            dA = dZ @ W[capa].T                    # por W.T ...
            dZ = dA * df(zs[capa - 1], activaciones[capa])   # ... y por la derivada
    return dW, db


def perdida(a, Y):
    a = np.clip(a, 1e-12, 1 - 1e-12)
    return -np.mean(Y * np.log(a) + (1 - Y) * np.log(1 - a))


X, y = generar_datos_circulos()
Y = y.reshape(-1, 1)
TAMANOS = [2, 16, 16, 16, 16, 16, 16, 1]
print(f"\nDatos: {X.shape}, red: {TAMANOS}")


# ============================================================
# PARTE 4: GRADIENTES QUE DESAPARECEN
# ============================================================
# Con los pesos iniciales, medimos el tamano (norma) de dW en cada capa.
# Cada vez que el delta retrocede una capa se multiplica por la derivada
# de la activacion. Si esa derivada es chica (sigmoid: maximo 0.25), el
# gradiente se achica capa tras capa.

print("\n--- Norma del gradiente por capa, con los pesos iniciales ---")
normas_por_capa = {}
for nombre in ACTIVACIONES:
    np.random.seed(0)
    W, b = inicializar_red(TAMANOS, nombre)
    zs, acts = forward(X, W, b, nombre)
    dW, _ = backward(Y, W, zs, acts, nombre)
    normas_por_capa[nombre] = [np.linalg.norm(g) for g in dW]
    print(f"{nombre:8s}", "  ".join(f"{v:.1e}" for v in normas_por_capa[nombre]))
for nombre in ACTIVACIONES:
    v = normas_por_capa[nombre]
    print(f"{nombre:8s} capa 1 / capa 7 = {v[0] / v[-1]:.1e}")


# ============================================================
# PARTE 5: ENTRENAR LA RED PROFUNDA CON CADA ACTIVACION
# ============================================================

def entrenar(nombre, tasa_aprendizaje=0.1, epocas=1000):
    np.random.seed(0)                              # misma semilla para comparar
    W, b = inicializar_red(TAMANOS, nombre)
    historial = []
    for epoca in range(epocas):
        zs, acts = forward(X, W, b, nombre)
        historial.append(perdida(acts[-1], Y))
        dW, db = backward(Y, W, zs, acts, nombre)
        for capa in range(len(W)):
            W[capa] -= tasa_aprendizaje * dW[capa]
            b[capa] -= tasa_aprendizaje * db[capa]
    accuracy = np.mean((forward(X, W, b, nombre)[1][-1] >= 0.5) == Y)
    return W, b, historial, accuracy


print("\n--- Entrenando la red de 6 capas ocultas (1000 epocas) ---")
resultados = {}
for nombre in ACTIVACIONES:
    resultados[nombre] = entrenar(nombre)
    hist = resultados[nombre][2]
    print(f"{nombre:8s} perdida: inicio={hist[0]:.4f}  epoca 500={hist[500]:.4f}  "
          f"final={hist[-1]:.4f} | accuracy={resultados[nombre][3]:.2%}")

# Neuronas ReLU "muertas": nunca se activan con ningun dato
W_r, b_r = resultados["relu"][0], resultados["relu"][1]
_, acts_r = forward(X, W_r, b_r, "relu")
muertas = sum(int(np.sum(np.all(a <= 0, axis=0))) for a in acts_r[1:-1])
print(f"\nNeuronas ReLU muertas despues de entrenar: {muertas} de {16 * 6}")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================

COLORES = {"step": "#9CA3AF", "sigmoid": "#3B82F6", "tanh": "#F59E0B", "relu": "#10B981"}
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
z = np.linspace(-4, 4, 400)

for nombre, (f, df) in [("step", (step, step_derivada))] + list(ACTIVACIONES.items()):
    a = f(z)
    ax1.plot(z, a, color=COLORES[nombre], label=nombre, linewidth=2)
    ax2.plot(z, df(z, a), color=COLORES[nombre], label=nombre, linewidth=2)
ax1.set_title("Activaciones")
ax1.set_xlabel("z")
ax1.set_ylim(-1.2, 2.2)
ax1.axhline(0, color="black", linewidth=0.5)
ax1.legend()
ax2.set_title("Derivadas")
ax2.set_xlabel("z")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend()

for nombre, normas in normas_por_capa.items():
    ax3.plot(range(1, len(normas) + 1), normas, marker="o", color=COLORES[nombre], label=nombre)
ax3.set_yscale("log")
ax3.set_title("Gradiente por capa al inicio (6 capas ocultas)")
ax3.set_xlabel("Capa (1 = la mas cercana a la entrada)")
ax3.set_ylabel("Norma de dW (escala log)")
ax3.legend()

for nombre in ACTIVACIONES:
    ax4.plot(resultados[nombre][2], color=COLORES[nombre],
             label=f"{nombre} ({resultados[nombre][3]:.0%})")
ax4.set_title("Entrenamiento de la red profunda")
ax4.set_xlabel("Epoca")
ax4.set_ylabel("Perdida (cross-entropy)")
ax4.legend()

plt.tight_layout()
plt.savefig("06_activation_functions.png", dpi=120)
print("\nGrafico guardado en: 06_activation_functions.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Sin activacion no lineal, apilar capas no agrega nada: X @ W1 @ W2 es
   una sola capa X @ (W1 @ W2)
2. Step fue la activacion del perceptron original, pero su derivada es 0:
   no sirve para backprop
3. Sigmoid (0 a 1) y tanh (-1 a 1) se saturan; la derivada de sigmoid
   es como maximo 0.25 y achica el gradiente en cada capa
4. ReLU = max(0, z): derivada 1 para z > 0, asi el gradiente pasa entero.
   Es la opcion por defecto para capas ocultas
5. ReLU puede "morir": si z es negativo para todos los datos, la neurona
   no recibe gradiente y deja de aprender

EXPERIMENTOS SUGERIDOS:
- Cambia TAMANOS a [2, 16, 1] (una capa oculta): la diferencia entre
  sigmoid y ReLU sigue siendo tan grande?
- Inicializa ReLU con sqrt(1/n) en vez de sqrt(2/n): que pasa con la
  norma del gradiente en las primeras capas?
- Sube tasa_aprendizaje a 1.0 en entrenar(): aumentan las neuronas ReLU muertas?
""")
