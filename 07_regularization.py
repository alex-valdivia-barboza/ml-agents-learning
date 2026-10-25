"""
LECCION 1.6 - REGULARIZACION
============================

Objetivo: ver el overfitting (la red memoriza los datos de entrenamiento
en vez de aprender el patron general) y tres tecnicas para combatirlo:
L2, L1 y dropout. Tambien early stopping: frenar en el momento justo.

En la leccion 1.4 la diferencia entre train y test fue de solo 1 punto,
porque habia 60 000 imagenes. Aqui provocamos el problema a proposito:
  - pocos datos (100 puntos de entrenamiento) y con mucho ruido
  - una red grande (2 -> 100 -> 100 -> 1, ReLU): 10 501 parametros
Con 105 parametros por punto, la red puede memorizar cada punto,
incluidos los que el ruido dejo del lado equivocado.

Usamos 3 conjuntos (ver fundamento de train y test de la leccion 1.4):
  - train (100):      para aprender
  - validacion (400): para comparar tecnicas y elegir cuando frenar
  - test (400):       se mira UNA vez al final, con la tecnica elegida
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: DATOS (circulos con mucho ruido)
# ============================================================
# Mismo generador de la 1.2, con dispersion 0.45 en vez de 0.15: las dos
# clases se mezclan en la zona del medio. Ningun modelo puede acertar el
# 100 % en datos nuevos; el limite lo pone el ruido, no la red.

def generar_datos_circulos(n_por_clase, dispersion=0.45):
    angulos = np.random.uniform(0, 2 * np.pi, 2 * n_por_clase)
    radios = np.hstack([np.random.randn(n_por_clase) * dispersion + 0.5,
                        np.random.randn(n_por_clase) * dispersion + 1.5])
    X = np.column_stack([radios * np.cos(angulos), radios * np.sin(angulos)])
    y = np.hstack([np.zeros(n_por_clase), np.ones(n_por_clase)])
    indices = np.random.permutation(len(X))
    return X[indices], y[indices].reshape(-1, 1)


X_train, Y_train = generar_datos_circulos(50)       # 100 puntos
X_val, Y_val = generar_datos_circulos(200)          # 400 puntos
X_test, Y_test = generar_datos_circulos(200)        # 400 puntos
print(f"Train: {X_train.shape}  Validacion: {X_val.shape}  Test: {X_test.shape}")


# ============================================================
# PARTE 2: LA RED (igual que la 1.5, con ReLU) + DROPOUT
# ============================================================
# Dropout: durante el entrenamiento, en cada epoca se "apaga" al azar una
# fraccion de las neuronas ocultas (su salida se multiplica por 0). Las
# que quedan se dividen por la probabilidad de quedar (1 - dropout), asi
# el valor esperado de la capa no cambia. Al predecir NO se apaga nada.

TAMANOS = [2, 100, 100, 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def inicializar_red(semilla=0):
    rng = np.random.default_rng(semilla)          # generador propio: misma inicializacion para todas las tecnicas
    W = [rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
         for n_in, n_out in zip(TAMANOS[:-1], TAMANOS[1:])]
    b = [np.zeros(n_out) for n_out in TAMANOS[1:]]
    return W, b


def forward(X, W, b, dropout=0.0, rng=None):
    """Si dropout > 0 (solo al entrenar), apaga neuronas ocultas al azar."""
    zs, acts, mascaras = [], [X], []
    for capa in range(len(W)):
        z = acts[-1] @ W[capa] + b[capa]
        if capa == len(W) - 1:
            a = sigmoid(z)
        else:
            a = np.maximum(0, z)
            if dropout > 0:
                mascara = (rng.random(a.shape) >= dropout) / (1 - dropout)   # 0 o 1/(1-p)
                a = a * mascara
                mascaras.append(mascara)
        zs.append(z)
        acts.append(a)
    return zs, acts, mascaras


def backward(Y, W, zs, acts, mascaras):
    n = len(Y)
    dW, db = [None] * len(W), [None] * len(W)
    dZ = (acts[-1] - Y) / n
    for capa in reversed(range(len(W))):
        dW[capa] = acts[capa].T @ dZ
        db[capa] = dZ.sum(axis=0)
        if capa > 0:
            dA = dZ @ W[capa].T
            if mascaras:
                dA = dA * mascaras[capa - 1]           # las neuronas apagadas no reciben gradiente
            dZ = dA * (zs[capa - 1] > 0)               # derivada de ReLU
    return dW, db


def perdida(X, Y, W, b):
    """Cross-entropy binaria (sin el termino de regularizacion)."""
    a = np.clip(forward(X, W, b)[1][-1], 1e-12, 1 - 1e-12)
    return -np.mean(Y * np.log(a) + (1 - Y) * np.log(1 - a))


def accuracy(X, Y, W, b):
    return np.mean((forward(X, W, b)[1][-1] >= 0.5) == Y)


# ============================================================
# PARTE 3: ENTRENAMIENTO CON REGULARIZACION
# ============================================================
# L2: a la perdida se le suma  l2/2 * suma(W^2)  -> gradiente extra: l2 * W
#     Cada paso "encoge" todos los pesos un poco, en proporcion a su tamano.
# L1: a la perdida se le suma  l1 * suma(|W|)    -> gradiente extra: l1 * signo(W)
#     Cada paso resta una cantidad fija: los pesos chicos llegan a 0.
# (Los sesgos b no se regularizan: solo mueven la frontera, no la "doblan".)

def entrenar(l2=0.0, l1=0.0, dropout=0.0, tasa_aprendizaje=0.3, epocas=8000):
    W, b = inicializar_red()
    rng = np.random.default_rng(1)
    historial = {"train": [], "val": []}
    for epoca in range(epocas):
        zs, acts, mascaras = forward(X_train, W, b, dropout, rng)
        dW, db = backward(Y_train, W, zs, acts, mascaras)
        for capa in range(len(W)):
            dW[capa] += l2 * W[capa] + l1 * np.sign(W[capa])
            W[capa] -= tasa_aprendizaje * dW[capa]
            b[capa] -= tasa_aprendizaje * db[capa]
        historial["train"].append(perdida(X_train, Y_train, W, b))
        historial["val"].append(perdida(X_val, Y_val, W, b))
    return W, b, historial


TECNICAS = {
    "Sin regularizacion": {},
    "L2 (0.01)": {"l2": 0.01},
    "L1 (0.003)": {"l1": 0.003},
    "Dropout (0.5)": {"dropout": 0.5},
}

print("\n--- Entrenando con cada tecnica (8000 epocas, full-batch) ---")
resultados = {}
for nombre, opciones in TECNICAS.items():
    W, b, hist = entrenar(**opciones)
    mejor_epoca = int(np.argmin(hist["val"]))
    pesos = np.hstack([w.ravel() for w in W])
    resultados[nombre] = {"W": W, "b": b, "hist": hist, "mejor_epoca": mejor_epoca, "pesos": pesos}
    print(f"{nombre:20s} | acc train={accuracy(X_train, Y_train, W, b):.1%}  "
          f"val={accuracy(X_val, Y_val, W, b):.1%} | perdida train={hist['train'][-1]:.3f}  "
          f"val={hist['val'][-1]:.3f} | |w| medio={np.abs(pesos).mean():.3f}  "
          f"pesos ~0: {np.mean(np.abs(pesos) < 1e-3):.0%}")


# ============================================================
# PARTE 4: EARLY STOPPING Y EVALUACION FINAL EN TEST
# ============================================================
# Early stopping: en vez de entrenar las 8000 epocas, quedarse con los
# pesos de la epoca donde la perdida de VALIDACION fue minima.
# Elegimos la tecnica con menor perdida de validacion, y solo entonces
# miramos el test, una unica vez.

sin_reg = resultados["Sin regularizacion"]
print(f"\nSin regularizacion: la perdida de validacion fue minima en la epoca "
      f"{sin_reg['mejor_epoca']} ({sin_reg['hist']['val'][sin_reg['mejor_epoca']]:.3f}) "
      f"y termino en {sin_reg['hist']['val'][-1]:.3f}")

mejor = min(resultados, key=lambda k: resultados[k]["hist"]["val"][-1])
W_m, b_m = resultados[mejor]["W"], resultados[mejor]["b"]
print(f"Tecnica elegida por validacion: {mejor}")
print(f"Accuracy en TEST (una sola vez): {accuracy(X_test, Y_test, W_m, b_m):.1%}")
print(f"(Sin regularizacion, en test:     {accuracy(X_test, Y_test, sin_reg['W'], sin_reg['b']):.1%})")


# ============================================================
# PARTE 5: VISUALIZACION
# ============================================================
# Fila 1: frontera de decision de cada tecnica sobre los 100 puntos de train
# Fila 2: (a) train vs validacion sin regularizacion, (b) validacion de
#         las 4 tecnicas, (c) distribucion de los pesos

fig, ejes = plt.subplots(2, 4, figsize=(18, 9))
xx, yy = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-3, 3, 250))
grilla = np.column_stack([xx.ravel(), yy.ravel()])
COLORES = ["#EF4444", "#3B82F6", "#8B5CF6", "#10B981"]

for ax, (nombre, r) in zip(ejes[0], resultados.items()):
    prob = forward(grilla, r["W"], r["b"])[1][-1].reshape(xx.shape)
    ax.contourf(xx, yy, prob, levels=[0, 0.5, 1], colors=["#DBEAFE", "#FEE2E2"], alpha=0.8)
    ax.contour(xx, yy, prob, levels=[0.5], colors="black", linewidths=1)
    m0, m1 = Y_train.ravel() == 0, Y_train.ravel() == 1
    ax.scatter(X_train[m0, 0], X_train[m0, 1], color="#3B82F6", edgecolor="white", s=30)
    ax.scatter(X_train[m1, 0], X_train[m1, 1], color="#EF4444", edgecolor="white", s=30)
    ax.set_title(f"{nombre}\nval={accuracy(X_val, Y_val, r['W'], r['b']):.0%}")
    ax.set_aspect("equal")

h = sin_reg["hist"]
ejes[1, 0].plot(h["train"], color="#3B82F6", label="Train")
ejes[1, 0].plot(h["val"], color="#F59E0B", label="Validacion")
ejes[1, 0].axvline(sin_reg["mejor_epoca"], color="gray", linestyle="--", label="Early stopping")
ejes[1, 0].set_title("Sin regularizacion: train vs validacion")
ejes[1, 0].set_xlabel("Epoca")
ejes[1, 0].set_ylabel("Perdida")
ejes[1, 0].legend()

for color, (nombre, r) in zip(COLORES, resultados.items()):
    ejes[1, 1].plot(r["hist"]["val"], color=color, label=nombre)
ejes[1, 1].set_title("Perdida de validacion")
ejes[1, 1].set_xlabel("Epoca")
ejes[1, 1].legend()

for color, (nombre, r) in zip(COLORES, resultados.items()):
    if nombre != "Dropout (0.5)":
        ejes[1, 2].hist(np.abs(r["pesos"]), bins=60, range=(0, 1.5), histtype="step",
                        color=color, label=nombre, linewidth=1.5)
ejes[1, 2].set_yscale("log")
ejes[1, 2].set_title("Distribucion de |pesos|")
ejes[1, 2].set_xlabel("|w|")
ejes[1, 2].legend()

ejes[1, 3].axis("off")
plt.tight_layout()
plt.savefig("07_regularization.png", dpi=110)
print("\nGrafico guardado en: 07_regularization.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Overfitting: la perdida de train sigue bajando mientras la de
   validacion sube. La red memoriza ruido en vez de aprender el patron
2. Validacion: un tercer conjunto para comparar tecnicas y elegir
   hiperparametros sin "gastar" el test
3. L2 suma l2*W al gradiente: encoge todos los pesos -> fronteras suaves
4. L1 suma l1*signo(W): lleva muchos pesos exactamente a 0
5. Dropout apaga neuronas al azar al entrenar: la red no puede depender
   de ninguna neurona concreta
6. Early stopping: quedarse con los pesos de la epoca de menor perdida
   de validacion

EXPERIMENTOS SUGERIDOS:
- Prueba l2 = 0.001 y l2 = 0.1: cual es demasiado poco y cual demasiado?
- Genera 1000 puntos de train en vez de 100: sigue habiendo overfitting
  sin regularizacion?
- Cambia dropout a 0.2 y a 0.8
""")
