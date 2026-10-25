"""
LECCION 1.3 - BACKPROPAGATION COMPLETA
======================================

Objetivo: entender de donde sale cada linea del entrenamiento de la red
de 2 capas de la leccion 1.2.

En la 1.2 usamos estas formulas "de confianza":
  delta2 = error * y_pred * (1 - y_pred)
  delta1 = (delta2 @ W2.T) * a1 * (1 - a1)
Aqui las derivamos con la REGLA DE LA CADENA, capa por capa, y despues
COMPROBAMOS que son correctas comparandolas con una aproximacion numerica
(gradient checking).

Misma red (2 -> 4 -> 1, sigmoid) y mismos datos XOR que la leccion 1.2.
Lo que cambia es la notacion, que ahora es la estandar:
  - una funcion de perdida L explicita
  - gradientes dL/dW (se nombran dW1, db1, dW2, db2)
  - actualizacion con signo menos: W = W - tasa * dW
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: DATOS (los mismos de la leccion 1.2)
# ============================================================

def generar_datos_xor(n_por_nube=50, dispersion=0.15):
    centros = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    etiquetas = [0, 0, 1, 1]
    nubes = [np.random.randn(n_por_nube, 2) * dispersion + c for c in centros]
    X = np.vstack(nubes)
    y = np.hstack([np.full(n_por_nube, e, dtype=float) for e in etiquetas])
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


X, y = generar_datos_xor()
Y = y.reshape(-1, 1)                    # (200, 1), como en la 1.2
print(f"Datos: X.shape={X.shape}, Y.shape={Y.shape}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# PARTE 2: LA REGLA DE LA CADENA CON UN SOLO NUMERO
# ============================================================
# Antes de la red completa, una cadena minima: una neurona, un punto.
#   z = w * x + b      ->  a = sigmoid(z)  ->  L = 1/2 * (a - y)^2
# Queremos dL/dw: "si muevo w un poquito, cuanto cambia L".
# La regla de la cadena dice que se multiplican las derivadas de cada
# eslabon:
#   dL/dw = dL/da * da/dz * dz/dw
#         = (a - y) * a*(1-a) * x

print("\n--- Regla de la cadena con un solo numero ---")
x_ej, y_ej, w_ej, b_ej = 1.5, 1.0, 0.8, -0.5
z_ej = w_ej * x_ej + b_ej
a_ej = sigmoid(z_ej)
L_ej = 0.5 * (a_ej - y_ej) ** 2

dL_da = a_ej - y_ej                     # eslabon 1: derivada de 1/2*(a-y)^2
da_dz = a_ej * (1 - a_ej)               # eslabon 2: derivada de sigmoid
dz_dw = x_ej                            # eslabon 3: derivada de w*x+b respecto a w
dL_dw = dL_da * da_dz * dz_dw           # la cadena: se multiplican
print(f"z={z_ej:.4f}  a={a_ej:.4f}  L={L_ej:.6f}")
print(f"dL/da={dL_da:.4f}  da/dz={da_dz:.4f}  dz/dw={dz_dw:.4f}  ->  dL/dw={dL_dw:.6f}")

# Comprobacion: movemos w un poquito (h) y medimos cuanto cambia L de verdad
h = 1e-5
L_mas = 0.5 * (sigmoid((w_ej + h) * x_ej + b_ej) - y_ej) ** 2
L_menos = 0.5 * (sigmoid((w_ej - h) * x_ej + b_ej) - y_ej) ** 2
print(f"Aproximacion numerica (L(w+h) - L(w-h)) / 2h = {(L_mas - L_menos) / (2 * h):.6f}")


# ============================================================
# PARTE 3: FORWARD PASS QUE GUARDA LOS VALORES INTERMEDIOS
# ============================================================
# Backprop necesita z1, a1 y a2 de la pasada hacia adelante: las
# derivadas de sigmoid se calculan con ellos. Por eso el forward los
# guarda en un diccionario "cache".

def inicializar_red(n_entrada=2, n_oculta=4, n_salida=1):
    W1 = np.random.randn(n_entrada, n_oculta)
    b1 = np.zeros(n_oculta)
    W2 = np.random.randn(n_oculta, n_salida)
    b2 = np.zeros(n_salida)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]          # (n, 4)
    a1 = sigmoid(z1)                    # (n, 4)
    z2 = a1 @ p["W2"] + p["b2"]         # (n, 1)
    a2 = sigmoid(z2)                    # (n, 1)  = y_pred
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache


def perdida(a2, Y):
    """Error cuadratico medio con un 1/2 adelante: L = 1/2 * promedio((a2 - Y)^2).
    El 1/2 solo sirve para que al derivar el 2 del cuadrado se cancele."""
    return 0.5 * np.mean((a2 - Y) ** 2)


# ============================================================
# PARTE 4: BACKWARD PASS, CAPA POR CAPA
# ============================================================
# Recorremos la red al reves. En cada paso aplicamos un eslabon de la
# cadena. La notacion dX significa "dL/dX": cuanto cambia la perdida si
# X cambia un poquito. Cada dX tiene la MISMA shape que X.
#
#   L  <- a2 <- z2 <- (W2, b2, a1) <- z1 <- (W1, b1, X)

def backward(Y, p, cache):
    n = len(Y)
    a1, a2, X = cache["a1"], cache["a2"], cache["X"]

    # --- Capa de salida ---
    dA2 = (a2 - Y) / n                          # (n,1)  dL/da2: derivada de 1/2*promedio((a2-Y)^2)
    dZ2 = dA2 * a2 * (1 - a2)                   # (n,1)  x derivada de sigmoid  -> "delta" de la salida
    dW2 = a1.T @ dZ2                            # (4,n) @ (n,1) -> (4,1)  como W2
    db2 = np.sum(dZ2, axis=0)                   # (1,)                    como b2

    # --- Hacia la capa oculta ---
    dA1 = dZ2 @ p["W2"].T                       # (n,1) @ (1,4) -> (n,4)  cuanto le toca a cada neurona
    dZ1 = dA1 * a1 * (1 - a1)                   # (n,4)  x derivada de sigmoid  -> "delta" oculto
    dW1 = X.T @ dZ1                             # (2,n) @ (n,4) -> (2,4)  como W1
    db1 = np.sum(dZ1, axis=0)                   # (4,)                    como b1

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ============================================================
# PARTE 5: GRADIENT CHECKING (comprobar que backward no tiene errores)
# ============================================================
# Para cada uno de los 17 parametros: lo subimos h, medimos L; lo bajamos
# h, medimos L. La pendiente (L+ - L-) / 2h aproxima dL/dparametro sin
# usar ninguna formula de derivadas. Si backward esta bien, ambos numeros
# coinciden en muchos decimales.

def gradiente_numerico(X, Y, p, h=1e-5):
    grads = {}
    for nombre, matriz in p.items():
        g = np.zeros_like(matriz)
        for idx in np.ndindex(matriz.shape):    # recorre cada posicion de la matriz
            original = matriz[idx]
            matriz[idx] = original + h
            L_mas = perdida(forward(X, p)[0], Y)
            matriz[idx] = original - h
            L_menos = perdida(forward(X, p)[0], Y)
            matriz[idx] = original              # lo dejamos como estaba
            g[idx] = (L_mas - L_menos) / (2 * h)
        grads[nombre] = g
    return grads


print("\n--- Gradient checking con los pesos iniciales ---")
params = inicializar_red()
a2, cache = forward(X, params)
grad_backprop = backward(Y, params, cache)
grad_numerico = gradiente_numerico(X, Y, params)

for nombre in ["W1", "b1", "W2", "b2"]:
    gb, gn = grad_backprop[nombre].ravel(), grad_numerico[nombre].ravel()
    error_relativo = np.linalg.norm(gb - gn) / (np.linalg.norm(gb) + np.linalg.norm(gn))
    print(f"{nombre}: backprop={np.round(gb, 6)}")
    print(f"{'':4}numerico={np.round(gn, 6)}  error relativo={error_relativo:.1e}")


# ============================================================
# PARTE 6: ENTRENAMIENTO CON W = W - tasa * dW
# ============================================================
# Es el mismo entrenamiento de la 1.2, escrito con la notacion estandar.
# Como dW2 = -(a1.T @ delta2 / n) de la 1.2, restar dW es lo mismo que
# sumar el gradiente de antes: es la misma cuenta con otro nombre.

def entrenar(X, Y, params, tasa_aprendizaje=2.0, epocas=3000):
    historial_perdida = []
    historial_norma = {"W1": [], "W2": []}
    for epoca in range(epocas):
        a2, cache = forward(X, params)                  # 1. forward (guarda cache)
        historial_perdida.append(perdida(a2, Y))        # 2. medir la perdida
        grads = backward(Y, params, cache)              # 3. backward
        for nombre in params:                           # 4. actualizar los 4 parametros
            params[nombre] -= tasa_aprendizaje * grads[nombre]
        historial_norma["W1"].append(np.linalg.norm(grads["W1"]))
        historial_norma["W2"].append(np.linalg.norm(grads["W2"]))
        if epoca % 500 == 0 or epoca == epocas - 1:
            accuracy = np.mean((a2 >= 0.5) == Y)
            print(f"Epoca {epoca:4d} | perdida={historial_perdida[-1]:.5f} | accuracy={accuracy:.2%}")
    return params, historial_perdida, historial_norma


print("\n--- Entrenando con backprop ---")
params, historial_perdida, historial_norma = entrenar(X, Y, params)
accuracy_final = np.mean((forward(X, params)[0] >= 0.5) == Y)
print(f"\nAccuracy final: {accuracy_final:.2%}")
print(f"W1 final =\n{np.round(params['W1'], 2)}")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================
# (1) gradient checking: cada punto es un parametro, backprop vs numerico
# (2) curva de la perdida
# (3) tamano (norma) de los gradientes de cada capa durante el entrenamiento

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

todos_b = np.hstack([grad_backprop[k].ravel() for k in ["W1", "b1", "W2", "b2"]])
todos_n = np.hstack([grad_numerico[k].ravel() for k in ["W1", "b1", "W2", "b2"]])
lim = np.abs(todos_b).max() * 1.1
ax1.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", label="iguales (diagonal)")
ax1.scatter(todos_n, todos_b, color="#3B82F6", edgecolor="white", s=60, label="17 parametros")
ax1.set_title("Gradient checking: backprop vs numerico")
ax1.set_xlabel("Gradiente numerico")
ax1.set_ylabel("Gradiente de backprop")
ax1.legend()

ax2.plot(historial_perdida, color="#10B981")
ax2.set_title("Perdida L durante el entrenamiento")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("L = 1/2 * MSE")

ax3.plot(historial_norma["W2"], color="#EF4444", label="||dW2|| (salida)")
ax3.plot(historial_norma["W1"], color="#8B5CF6", label="||dW1|| (oculta)")
ax3.set_yscale("log")
ax3.set_title("Tamano de los gradientes por capa")
ax3.set_xlabel("Epoca")
ax3.set_ylabel("Norma del gradiente (escala log)")
ax3.legend()

plt.tight_layout()
plt.savefig("04_backpropagation_detailed.png", dpi=120)
print("\nGrafico guardado en: 04_backpropagation_detailed.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Regla de la cadena: la derivada de una cadena de funciones es el
   producto de las derivadas de cada eslabon
2. El forward guarda los valores intermedios (cache) porque el backward
   los necesita para calcular las derivadas
3. Backward recorre la red al reves: dA2 -> dZ2 -> (dW2, db2) -> dA1 ->
   dZ1 -> (dW1, db1). Cada dX tiene la misma shape que X
4. Para pasar de una capa a la anterior se multiplica por W.T, y para
   atravesar una activacion se multiplica por su derivada
5. Gradient checking compara backprop con (L(p+h) - L(p-h)) / 2h para
   detectar errores en las formulas

EXPERIMENTOS SUGERIDOS:
- Introduce un error a proposito en backward() (por ejemplo, borra el
  "* a1 * (1 - a1)" de dZ1) y mira como lo detecta el gradient checking
- Cambia h en gradiente_numerico() a 1e-2 y a 1e-12: por que ambos
  extremos dan peor aproximacion?
- Agrega una segunda capa oculta (W3, b3) a forward() y backward(): el
  patron "por W.T, por la derivada" se repite igual
""")
