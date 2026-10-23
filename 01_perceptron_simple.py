"""
LECCION 1.1 - PERCEPTRON SIMPLE
================================

Objetivo: entender la neurona artificial mas basica: el perceptron.

Un perceptron es una sola neurona que:
  1. Recibe entradas X (un vector de numeros)
  2. Las combina con pesos w y un sesgo (bias) b -> z = X . w + b
  3. Aplica una funcion de activacion sigmoid -> y = sigma(z)
  4. Compara y con la respuesta correcta y ajusta w y b si se equivoco

Vamos a entrenarlo para separar dos grupos de puntos en 2D que son
linealmente separables (se pueden dividir con una linea recta).
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# PARTE 1: GENERAR DATOS
# ============================================================
# Creamos dos "nubes" de puntos en 2D. Cada punto tiene 2 features
# (coordenada x1, x2) y una etiqueta (0 o 1) segun a que grupo pertenece.
# El factor 0.6 hace que los puntos se dispersen alrededor del centro de cada nube. mas chico el número = mas concentrados los puntos, mas grande = mas dispersos
# La funcion random.randn() genera numeros aleatorios con distribucion normal (campana de Gauss) centrada en 0 y desviacion estandar 1. Al multiplicar por 0.6 y sumar un vector, 
# desplazamos la nube a la posicion deseada.
# np.array([1.5, 1.5]) Recibe un solo argumento: una lista [1.5, 1.5], que np.array() convierte en un vector de 2 elementos. Al sumarse a la matriz de (50, 2), 
# NumPy usa broadcasting: el vector de 2 valores se aplica a cada una de las 50 filas, sumando 1.5 a la columna x1 y 1.5 a la columna x2 de cada punto. 
# Esto desplaza el centro de la nube de (0,0) a (1.5, 1.5).
# Mismo mecanismo para clase_1, pero centrada en (4.0, 4.0) en vez de (1.5, 1.5).
# El porqué de estos números específicos: los dos centros están a una distancia de ~3.5 unidades entre sí, mientras que el desvío de cada nube es solo 0.6 — por eso las nubes casi no se superponen, 
# y el perceptrón puede separarlas con una línea recta (dataset linealmente separable, que es justo lo que necesita la lección 1.1).

def generar_datos_linealmente_separables(n_por_clase=50):
    """Genera 2 nubes de puntos separables por una linea recta."""
    clase_0 = np.random.randn(n_por_clase, 2) * 0.6 + np.array([1.5, 1.5]) # genera coordenadas x1,x2 para la clase 0 centradas en (1.5, 1.5)
    clase_1 = np.random.randn(n_por_clase, 2) * 0.6 + np.array([4.0, 4.0]) # genera coordenadas x1,x2 para la clase 1 centradas en (4.0, 4.0)

    X = np.vstack([clase_0, clase_1])          # (100, 2) apilamos las dos nubes de puntos en una sola matriz de 100 filas y 2 columnas
    y = np.hstack([np.zeros(n_por_clase), np.ones(n_por_clase)])  # (100,) concatenamos las etiquetas: 50 ceros para clase 0 y 50 unos para clase 1

    # Mezclamos el orden para que el entrenamiento no vea siempre
    # primero una clase y despues la otra
    indices = np.random.permutation(len(X)) # retorna un array de indices aleatorios de 0 a 99 (len(X)=100)
    return X[indices], y[indices]           # retornamos X e y mezclados en el mismo orden de indices aleatorios para que conserven los pares (X[i], y[i]) correctos y no se pierda la correspondencia entre las coordenadas y su etiqueta.


X, y = generar_datos_linealmente_separables()
print(f"Datos generados: X.shape={X.shape}, y.shape={y.shape}")
print(f"Ejemplo de fila: X[0]={X[0]}, y[0]={y[0]}")


# ============================================================
# PARTE 2: LA NEURONA (forward pass)
# ============================================================
# z = X . w + b   (combinacion lineal de las entradas)
# y_pred = sigmoid(z)  (aplastamos z a un rango entre 0 y 1)
#
# sigmoid(z) = 1 / (1 + e^-z)
# Cuando z es muy positivo, sigmoid(z) -> 1
# Cuando z es muy negativo, sigmoid(z) -> 0
# Cuando z = 0, sigmoid(z) = 0.5 (el punto de decision)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(X, w, b):
    """Calcula la prediccion de la neurona para cada fila de X."""
    z = X @ w + b          # combinacion lineal: (n,2) @ (2,) + escalar -> (n,)
    y_pred = sigmoid(z)
    return y_pred


# ============================================================
# PARTE 3: ENTRENAMIENTO (backward pass / regla del perceptron)
# ============================================================
# Para cada ejemplo:
#   error = y_real - y_pred
#   Si el error es grande, ajustamos mucho los pesos.
#   Si el error es chico (predijo bien), casi no los tocamos.
#
# Regla de actualizacion (descenso de gradiente sobre error cuadratico
# con activacion sigmoid):
#   w = w + tasa_aprendizaje * error * derivada_sigmoid(z) * X
#   b = b + tasa_aprendizaje * error * derivada_sigmoid(z)
#
# derivada_sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
# Esta derivada nos dice "que tan sensible es la salida a cambios en z"

def entrenar_perceptron(X, y, tasa_aprendizaje=0.5, epocas=200):
    n_features = X.shape[1]
    w = np.random.randn(n_features) * 0.1   # pesos iniciales pequenos y aleatorios
    b = 0.0

    historial_error = []

    for epoca in range(epocas):
        y_pred = forward(X, w, b)
        error = y - y_pred                       # que tan lejos estamos de la respuesta correcta
        derivada = y_pred * (1 - y_pred)          # derivada de sigmoid en el punto actual

        # Gradientes: promedio sobre todos los ejemplos del dataset
        grad_w = X.T @ (error * derivada) / len(X)
        grad_b = np.mean(error * derivada)

        # Actualizamos pesos en la direccion que reduce el error
        w += tasa_aprendizaje * grad_w
        b += tasa_aprendizaje * grad_b

        error_cuadratico_medio = np.mean(error ** 2)
        historial_error.append(error_cuadratico_medio)

        if epoca % 40 == 0 or epoca == epocas - 1:
            predicciones_correctas = np.sum((y_pred >= 0.5).astype(int) == y)
            accuracy = predicciones_correctas / len(y)
            print(f"Epoca {epoca:3d} | error={error_cuadratico_medio:.4f} | accuracy={accuracy:.2%}")

    return w, b, historial_error


print("\n--- Entrenando el perceptron ---")
w_final, b_final, historial = entrenar_perceptron(X, y)
print(f"\nPesos finales: w={w_final}, b={b_final:.4f}")


# ============================================================
# PARTE 4: EVALUACION
# ============================================================

y_pred_final = forward(X, w_final, b_final)
y_pred_clase = (y_pred_final >= 0.5).astype(int)
accuracy_final = np.mean(y_pred_clase == y)
print(f"Accuracy final sobre datos de entrenamiento: {accuracy_final:.2%}")


# ============================================================
# PARTE 5: VISUALIZACION
# ============================================================
# Graficamos: (1) los datos + la linea de decision aprendida,
#             (2) la curva de aprendizaje (error vs epoca)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Grafico 1: datos + frontera de decision ---
ax1.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="#3B82F6", label="Clase 0", edgecolor="white")
ax1.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="#EF4444", label="Clase 1", edgecolor="white")

# La frontera de decision es donde z = 0, es decir: x1*w0 + x2*w1 + b = 0
# despejando x2: x2 = -(w0*x1 + b) / w1
x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_vals = -(w_final[0] * x1_vals + b_final) / w_final[1]
ax1.plot(x1_vals, x2_vals, color="black", linestyle="--", label="Frontera de decision")

ax1.set_title("Datos y frontera de decision aprendida")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.legend()

# --- Grafico 2: curva de aprendizaje ---
ax2.plot(historial, color="#10B981")
ax2.set_title("Curva de aprendizaje")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("Error cuadratico medio")

plt.tight_layout()
plt.savefig("01_perceptron_simple.png", dpi=120)
print("\nGrafico guardado en: 01_perceptron_simple.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Una neurona combina entradas con pesos (w) y un sesgo (b): z = X.w + b
2. La funcion de activacion (sigmoid) convierte z en una probabilidad (0 a 1)
3. El error mide que tan lejos esta la prediccion de la respuesta correcta
4. El aprendizaje = ajustar w y b para reducir el error, iteracion a iteracion
5. Un solo perceptron SOLO puede aprender fronteras LINEALES (una recta)
   -> en la leccion 1.2 veremos por que esto no alcanza para problemas
      como XOR, y como las capas ocultas resuelven esa limitacion

EXPERIMENTOS SUGERIDOS:
- Cambia `tasa_aprendizaje` a 0.01 y a 2.0: como afecta la velocidad
  y estabilidad del entrenamiento?
- Cambia `epocas` a 5: alcanza para converger?
- Modifica generar_datos_linealmente_separables() para que las nubes
  se superpongan mas: que le pasa al accuracy final?
""")
