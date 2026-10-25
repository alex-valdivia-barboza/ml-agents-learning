"""
LECCION 4.1 - AUTOGRAD: EL BACKWARD AUTOMATICO
==============================================

Objetivo: dejar de escribir el backward a mano.

En la 1.3 escribiste el backward de una red de 2 capas linea por linea
(dA2, dZ2, dW2...). En la 3.5 y la 3.6 lo hiciste otra vez para la RNN
y para la atencion, y cada vez hubo que verificarlo con gradient
checking porque es facil equivocarse. Un transformer tiene decenas de
operaciones: escribir su backward a mano no es practico.

La idea de AUTOGRAD (diferenciacion automatica):
  1. Cada operacion del forward ANOTA de donde salio su resultado.
     Asi el forward construye un GRAFO de calculo.
  2. Cada operacion sabe su propia derivada LOCAL (la suma pasa el
     gradiente igual, la multiplicacion lo multiplica por el otro
     factor, tanh por 1 - t^2...).
  3. backward() recorre el grafo al reves aplicando la regla de la
     cadena: gradiente que llega x derivada local.
Es lo que hacen PyTorch, JAX y TensorFlow. Aqui lo construimos con
escalares (clase Valor) y luego con arrays (mini_autograd.py).
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mini_autograd import Tensor, cross_entropy, parametros_en_cero

np.random.seed(42)


# ============================================================
# PARTE 1: UN NUMERO QUE RECUERDA DE DONDE VIENE
# ============================================================
# Un Valor guarda su numero (dato), su gradiente (grad), los Valores de
# los que salio (previos) y una funcion _atras que reparte el gradiente
# hacia ellos.

class Valor:
    def __init__(self, dato, previos=(), op="", nombre=""):
        self.dato = dato
        self.grad = 0.0
        self.previos = previos
        self.op = op
        self.nombre = nombre
        self._atras = lambda: None

    def __repr__(self):
        return f"Valor({self.nombre}={self.dato:.4f}, grad={self.grad:.4f})"

    def __add__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        salida = Valor(self.dato + otro.dato, (self, otro), "+")

        def _atras():
            self.grad += 1.0 * salida.grad             # d(a+b)/da = 1
            otro.grad += 1.0 * salida.grad             # d(a+b)/db = 1
        salida._atras = _atras
        return salida

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        salida = Valor(self.dato * otro.dato, (self, otro), "*")

        def _atras():
            self.grad += otro.dato * salida.grad       # d(a*b)/da = b
            otro.grad += self.dato * salida.grad       # d(a*b)/db = a
        salida._atras = _atras
        return salida

    def tanh(self):
        t = np.tanh(self.dato)
        salida = Valor(t, (self,), "tanh")

        def _atras():
            self.grad += (1 - t ** 2) * salida.grad    # derivada de tanh (leccion 1.5)
        salida._atras = _atras
        return salida

    def backward(self):
        # orden topologico: cada nodo aparece DESPUES de los nodos de los que depende
        orden, visitados = [], set()

        def construir(v):
            if v not in visitados:
                visitados.add(v)
                for p in v.previos:
                    construir(p)
                orden.append(v)
        construir(self)

        self.grad = 1.0                                # dL/dL = 1
        for v in reversed(orden):                      # de la salida hacia las entradas
            v._atras()


# Una neurona con 2 entradas: L = tanh(x1*w1 + x2*w2 + b)
x1, x2 = Valor(2.0, nombre="x1"), Valor(0.0, nombre="x2")
w1, w2 = Valor(-3.0, nombre="w1"), Valor(1.0, nombre="w2")
b = Valor(6.8813735870195432, nombre="b")
z = x1 * w1 + x2 * w2 + b
z.nombre = "z"
L = z.tanh()
L.nombre = "L"
L.backward()

print("PARTE 1: una neurona escalar")
print(f"  z = {z.dato:.4f}   L = tanh(z) = {L.dato:.4f}")
for v in [w1, w2, b, x1, x2]:
    print(f"  dL/d{v.nombre} = {v.grad:+.4f}")

# Comprobacion a mano con la regla de la cadena:
#   dL/dz = 1 - tanh(z)^2,  dL/dw1 = dL/dz * x1,  dL/dw2 = dL/dz * x2
dL_dz = 1 - L.dato ** 2
print(f"  A mano: dL/dz = {dL_dz:.4f}, dL/dw1 = {dL_dz * x1.dato:+.4f}, dL/dw2 = {dL_dz * x2.dato:+.4f}")


# ============================================================
# PARTE 2: POR QUE "+=": UN VALOR USADO DOS VECES
# ============================================================
# Si un valor se usa en dos lugares, recibe gradiente por los dos caminos
# y hay que SUMARLOS (es la regla de la cadena multivariable). Con "="
# el segundo camino pisaria al primero.

a = Valor(3.0, nombre="a")
f = a * a + a                                      # f = a^2 + a  ->  df/da = 2a + 1 = 7
f.backward()
print(f"\nPARTE 2: f = a*a + a con a = 3  ->  df/da = {a.grad} (esperado 2*3 + 1 = 7)")


# ============================================================
# PARTE 3: LO MISMO CON ARRAYS (mini_autograd.py)
# ============================================================
# Un Tensor es un Valor que guarda un array entero. Las reglas son las
# mismas, pero cada operacion de matrices tiene su derivada "en bloque"
# (la de X @ W es la de la 1.3: X.T @ delta). Lo comprobamos operacion
# por operacion contra el gradiente numerico (gradient checking, 1.3):
#     dL/dx ~ (L(x + h) - L(x - h)) / 2h

def gradiente_numerico(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        viejo = x[i]
        x[i] = viejo + h
        mas = f(x)
        x[i] = viejo - h
        menos = f(x)
        x[i] = viejo
        g[i] = (mas - menos) / (2 * h)
    return g


def comprobar(nombre, construir, forma):
    """construir(t) arma una expresion con el Tensor t y devuelve un escalar."""
    x = np.random.randn(*forma)
    t = Tensor(x.copy())
    construir(t).backward()
    num = gradiente_numerico(lambda arr: construir(Tensor(arr)).data.item(), x.copy())
    error = np.max(np.abs(t.grad - num)) / (np.max(np.abs(num)) + 1e-12)
    print(f"  {nombre:<28} error relativo = {error:.1e}")
    return error


M = np.random.randn(4, 3)
pesos_s = np.random.randn(5, 4)
print("\nPARTE 3: gradient checking de cada operacion")
pruebas = {
    "suma con broadcasting": (lambda t: ((t + Tensor(np.ones(3))) ** 2).sum(), (5, 3)),
    "producto elemento": (lambda t: (t * Tensor(M)).sum(), (4, 3)),
    "matmul": (lambda t: ((t @ Tensor(M)) ** 2).sum(), (5, 4)),
    "tanh": (lambda t: t.tanh().sum(), (3, 4)),
    "sigmoid": (lambda t: t.sigmoid().sum(), (3, 4)),
    "relu": (lambda t: (t.relu() * Tensor(M)).sum(), (4, 3)),
    "exp y log": (lambda t: (t.exp() + 1.0).log().sum(), (3, 4)),
    "mean por eje": (lambda t: (t.mean(axis=1) ** 2).sum(), (4, 3)),
    "softmax": (lambda t: (t.softmax() * Tensor(pesos_s)).sum(), (5, 4)),
    "reshape y transpose": (lambda t: (t.reshape(4, 3).transpose(1, 0) @ Tensor(M)).sum(), (2, 6)),
    "indexar filas (embedding)": (lambda t: (t[np.array([0, 2, 2, 1])] ** 2).sum(), (3, 4)),
    "cross_entropy": (lambda t: cross_entropy(t, np.array([1, 0, 3, 2, 2])), (5, 4)),
}
errores = {nombre: comprobar(nombre, f, forma) for nombre, (f, forma) in pruebas.items()}


# ============================================================
# PARTE 4: LA RED XOR DE LA 1.3, SIN ESCRIBIR EL BACKWARD
# ============================================================
# La misma red (2 -> 4 sigmoid -> 1 sigmoid), la misma perdida
# (1/2 * promedio del error al cuadrado) y la misma tasa. Solo el
# forward: los gradientes los calcula backward().

def generar_datos_xor(n_por_nube=50, dispersion=0.15):
    centros = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    etiquetas = [0, 0, 1, 1]
    nubes = [np.random.randn(n_por_nube, 2) * dispersion + c for c in centros]
    X = np.vstack(nubes)
    y = np.hstack([np.full(n_por_nube, e, dtype=float) for e in etiquetas])
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


X, y = generar_datos_xor()
Y = y.reshape(-1, 1)

W1 = Tensor(np.random.randn(2, 4), nombre="W1")
b1 = Tensor(np.zeros(4), nombre="b1")
W2 = Tensor(np.random.randn(4, 1), nombre="W2")
b2 = Tensor(np.zeros(1), nombre="b2")
parametros = [W1, b1, W2, b2]


def forward(X):
    a1 = (Tensor(X) @ W1 + b1).sigmoid()
    return (a1 @ W2 + b2).sigmoid()


def perdida(a2, Y):
    return ((a2 - Y) ** 2).mean() * 0.5


# El backward manual de la 1.3, para comparar en el primer paso
def backward_manual(X, Y):
    a1 = 1 / (1 + np.exp(-(X @ W1.data + b1.data)))
    a2 = 1 / (1 + np.exp(-(a1 @ W2.data + b2.data)))
    dZ2 = (a2 - Y) / len(Y) * a2 * (1 - a2)
    dZ1 = dZ2 @ W2.data.T * a1 * (1 - a1)
    return [X.T @ dZ1, dZ1.sum(axis=0), a1.T @ dZ2, dZ2.sum(axis=0)]


TASA, EPOCAS = 2.0, 3000
historial = []
inicio = time.time()
for epoca in range(EPOCAS):
    parametros_en_cero(parametros)                 # los gradientes se ACUMULAN: hay que limpiarlos
    a2 = forward(X)
    L = perdida(a2, Y)
    L.backward()
    if epoca == 0:
        manual = backward_manual(X, Y)
        dif = max(np.max(np.abs(p.grad - m)) for p, m in zip(parametros, manual))
        print(f"\nPARTE 4: diferencia autograd vs backward manual de la 1.3 = {dif:.1e}")
    for p in parametros:
        p.data -= TASA * p.grad
    historial.append(L.data.item())
    if epoca % 500 == 0:
        exactitud = np.mean((a2.data > 0.5) == Y)
        print(f"  Epoca {epoca:4d} | perdida={historial[-1]:.5f} | accuracy={exactitud:.2%}")
duracion = time.time() - inicio
exactitud = np.mean((forward(X).data > 0.5) == Y)
print(f"  Final: perdida={historial[-1]:.5f}, accuracy={exactitud:.2%}, {duracion:.2f} s")


# ============================================================
# PARTE 5: CUANTO CUESTA LA COMODIDAD
# ============================================================
# Autograd crea un objeto por operacion y recorre el grafo en Python.
# Medimos el tiempo de un paso (forward + backward) contra el manual.

def tiempo(fn, repeticiones=2000):
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        fn()
    return (time.perf_counter() - t0) / repeticiones * 1e6


def paso_auto():
    parametros_en_cero(parametros)
    perdida(forward(X), Y).backward()


t_auto = tiempo(paso_auto)
t_manual = tiempo(lambda: backward_manual(X, Y))
print(f"\nPARTE 5: un paso con autograd = {t_auto:.0f} us, manual = {t_manual:.0f} us "
      f"({t_auto / t_manual:.1f} veces mas lento)")


# ============================================================
# PARTE 6: GRAFICOS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

nombres = list(errores)
ax1.barh(nombres, [max(e, 1e-16) for e in errores.values()], color="#EC4899")
ax1.set_xscale("log")
ax1.axvline(1e-5, color="#6B7280", linestyle="--", linewidth=1)
ax1.text(1.3e-5, len(nombres) - 0.6, "umbral 1e-5", fontsize=8, color="#6B7280")
ax1.invert_yaxis()
ax1.set_title("Gradient checking: autograd vs numerico")
ax1.set_xlabel("Error relativo (escala log)")

ax2.plot(historial, color="#EC4899")
ax2.set_title("XOR con autograd (sin backward escrito a mano)")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("Perdida")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("19_autograd.png", dpi=120)
print("\nGrafico guardado en: 19_autograd.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. El forward construye un grafo: cada resultado recuerda de donde salio
2. Cada operacion solo conoce su derivada LOCAL; backward() las encadena
   recorriendo el grafo en orden topologico inverso
3. Los gradientes se SUMAN (+=) cuando un valor se usa mas de una vez;
   por eso hay que ponerlos en cero antes de cada paso
4. Con arrays, cada operacion tiene su derivada en bloque (X.T @ delta,
   suma sobre el eje del broadcasting...)
5. Da exactamente los mismos gradientes que el backward manual de la 1.3,
   a cambio de algo de velocidad

EXPERIMENTOS SUGERIDOS:
- Quita parametros_en_cero(parametros) del bucle: que pasa con la perdida?
- Cambia los "+=" de __mul__ en Valor por "=" y vuelve a calcular f = a*a + a
- Agrega a Tensor una operacion nueva (por ejemplo, sqrt) con su _atras,
  y compruebala con comprobar()
""")
