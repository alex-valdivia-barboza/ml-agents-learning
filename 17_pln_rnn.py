"""
LECCION 3.5 - REDES RECURRENTES (RNN)
=====================================

Objetivo: una red que lee un texto EN ORDEN, un caracter por vez, y
mantiene una MEMORIA (el estado oculto h) de lo que leyo hasta ahora.

Los n-gramas (3.3) miraban un contexto de largo fijo. Una RNN aplica la
misma capa en cada paso:

    h_t = tanh(Wxh @ x_t + Whh @ h_(t-1) + bh)     <- memoria actualizada
    y_t = Why @ h_t + by                          <- puntaje de cada caracter
    P(siguiente caracter) = softmax(y_t)

h_t depende de h_(t-1), que depende de h_(t-2)... asi, en teoria, la
prediccion puede depender de TODO lo anterior.

Trabajamos con CARACTERES (no palabras): el vocabulario es chico (~60
simbolos) y no hay palabras desconocidas. Inspirado en el min-char-rnn
de Andrej Karpathy.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from corpus_es import cargar_quijote

rng = np.random.default_rng(42)


# ============================================================
# PARTE 1: LOS DATOS (caracteres)
# ============================================================

texto = cargar_quijote().lower()
texto = " ".join(texto.split())                    # espacios y saltos de linea -> un espacio
caracteres = sorted(set(texto))
V = len(caracteres)
a_indice = {ch: i for i, ch in enumerate(caracteres)}
datos = np.array([a_indice[ch] for ch in texto])
corte = int(len(datos) * 0.95)
train, test = datos[:corte], datos[corte:]
print(f"Caracteres: {len(datos):,} | vocabulario: {V} simbolos")
print("Simbolos:", "".join(caracteres))


# ============================================================
# PARTE 2: LA RED
# ============================================================
# H = 128 neuronas de memoria. Pesos chicos al azar (escala de la 1.4).

H = 128
Wxh = rng.standard_normal((H, V)) * 0.01           # entrada -> memoria
Whh = rng.standard_normal((H, H)) * 0.01           # memoria anterior -> memoria nueva
Why = rng.standard_normal((V, H)) * 0.01           # memoria -> salida
bh, by = np.zeros(H), np.zeros(V)
params = [Wxh, Whh, Why, bh, by]


def forward_backward(entradas, objetivos, h_prev):
    """Forward sobre una secuencia y backward a traves del tiempo (BPTT)."""
    xs, hs, ps = {}, {-1: h_prev}, {}
    perdida = 0.0
    for t, (i, j) in enumerate(zip(entradas, objetivos)):
        xs[t] = np.zeros(V)
        xs[t][i] = 1.0                                         # one-hot del caracter actual
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)   # la memoria se actualiza
        y = Why @ hs[t] + by
        ps[t] = np.exp(y - y.max())
        ps[t] /= ps[t].sum()                                   # softmax (leccion 1.4)
        perdida += -np.log(ps[t][j])                           # cross-entropy

    grads = [np.zeros_like(p) for p in params]
    dWxh, dWhh, dWhy, dbh, dby = grads
    dh_siguiente = np.zeros(H)                                 # gradiente que llega desde el futuro
    for t in reversed(range(len(entradas))):
        dy = ps[t].copy()
        dy[objetivos[t]] -= 1                                  # probs - one_hot (leccion 1.4)
        dWhy += np.outer(dy, hs[t])
        dby += dy
        dh = Why.T @ dy + dh_siguiente                         # culpa por esta salida + por el futuro
        dz = (1 - hs[t] ** 2) * dh                             # derivada de tanh (leccion 1.5)
        dbh += dz
        dWxh += np.outer(dz, xs[t])
        dWhh += np.outer(dz, hs[t - 1])
        dh_siguiente = Whh.T @ dz                              # pasar la culpa al paso anterior
    for g in grads:
        np.clip(g, -5, 5, out=g)                               # recorte: evita gradientes que explotan
    return perdida / len(entradas), grads, hs[len(entradas) - 1]


def muestrear(h, primer_caracter, n, temperatura=1.0):
    """Genera n caracteres sorteando de la softmax, como generar() de la 3.3."""
    x = np.zeros(V)
    x[a_indice[primer_caracter]] = 1
    salida = [primer_caracter]
    for _ in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = (Why @ h + by) / temperatura
        p = np.exp(y - y.max())
        p /= p.sum()
        i = rng.choice(V, p=p)
        x = np.zeros(V)
        x[i] = 1
        salida.append(caracteres[i])
    return "".join(salida)


# ============================================================
# PARTE 3: ENTRENAMIENTO (Adagrad)
# ============================================================
# Se recorre el texto en tramos de 25 caracteres. La memoria h pasa de un
# tramo al siguiente, pero el gradiente se corta cada 25 pasos.
# Adagrad: cada peso tiene su propia tasa, que baja si ese peso ya
# recibio gradientes grandes (acumula los cuadrados en `memoria`).

LARGO, TASA, ITERACIONES = 25, 0.1, 80000
memoria = [np.zeros_like(p) for p in params]
perdida_suave = -np.log(1.0 / V)                   # perdida de adivinar al azar
historial = []
pos, h = 0, np.zeros(H)
inicio = time.time()
for it in range(ITERACIONES + 1):
    if pos + LARGO + 1 >= len(train):
        pos, h = 0, np.zeros(H)
    entradas = train[pos:pos + LARGO]
    objetivos = train[pos + 1:pos + LARGO + 1]                 # cada caracter predice el siguiente
    perdida, grads, h = forward_backward(entradas, objetivos, h)
    for p, g, m in zip(params, grads, memoria):
        m += g * g
        p -= TASA * g / np.sqrt(m + 1e-8)
    perdida_suave = 0.999 * perdida_suave + 0.001 * perdida
    historial.append(perdida_suave)
    if it % 20000 == 0:
        print(f"\nIteracion {it:5d} | perdida {perdida_suave:.3f} | {time.time() - inicio:.0f} s")
        print("  >", muestrear(np.zeros(H), "e", 120))
    pos += LARGO


# ============================================================
# PARTE 4: EVALUAR EN TEXTO NUEVO
# ============================================================
# Perplejidad por caracter (leccion 3.3) en el 5% final, comparada con
# un modelo de bigramas de caracteres (el caracter anterior y nada mas).

def perdida_test(datos):
    h, total = np.zeros(H), 0.0
    for i, j in zip(datos[:-1], datos[1:]):
        h = np.tanh(Wxh[:, i] + Whh @ h + bh)                 # Wxh @ one_hot = una columna de Wxh
        y = Why @ h + by
        y = y - y.max()
        total += -(y[j] - np.log(np.exp(y).sum()))
    return total / (len(datos) - 1)


cuentas = np.ones((V, V))                                      # bigramas de caracteres con Laplace
np.add.at(cuentas, (train[:-1], train[1:]), 1)
probs_bigrama = cuentas / cuentas.sum(axis=1, keepdims=True)
muestra = test[:20000]
ce_bigrama = -np.mean(np.log(probs_bigrama[muestra[:-1], muestra[1:]]))
ce_rnn = perdida_test(muestra)
print(f"\nPerplejidad por caracter en test: bigramas = {np.exp(ce_bigrama):.2f} | RNN = {np.exp(ce_rnn):.2f}")


# ============================================================
# PARTE 5: CUANTO "RECUERDA": EL GRADIENTE A TRAVES DEL TIEMPO
# ============================================================
# Se mide cuanto influye el caracter de hace k pasos en la prediccion
# actual: la norma del gradiente que llega k pasos atras desde la ultima
# salida. Cada paso hacia atras multiplica por Whh.T y por la derivada de
# tanh (como las capas de la leccion 1.5).

entradas = test[1000:1101]
hs = {-1: np.zeros(H)}
for t, i in enumerate(entradas):
    hs[t] = np.tanh(Wxh[:, i] + Whh @ hs[t - 1] + bh)
T = len(entradas) - 1
dy = np.zeros(V)
dy[test[1101]] = -1
dh = Why.T @ dy
normas = []
for t in range(T, T - 60, -1):
    dz = (1 - hs[t] ** 2) * dh
    normas.append(np.linalg.norm(dz))
    dh = Whh.T @ dz
print("Norma del gradiente k pasos atras:", " ".join(f"k={k}: {normas[k]:.1e}" for k in (0, 5, 10, 20, 40)))


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(historial, color="#F59E0B")
ax1.axhline(ce_bigrama, color="gray", linestyle="--", label=f"bigramas de caracteres (test)")
ax1.set_title("Perdida de entrenamiento (cross-entropy por caracter)")
ax1.set_xlabel("Iteracion (tramos de 25 caracteres)")
ax1.legend()

ax2.semilogy(range(len(normas)), normas, marker="o", markersize=3, color="#3B82F6")
ax2.set_title("Influencia del caracter de hace k pasos")
ax2.set_xlabel("k (pasos hacia atras)")
ax2.set_ylabel("Norma del gradiente (log)")

plt.tight_layout()
plt.savefig("17_pln_rnn.png", dpi=120)
print("\nGrafico guardado en: 17_pln_rnn.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Una RNN aplica la misma capa en cada paso y arrastra un estado oculto h:
   la memoria de lo leido
2. Se entrena con backpropagation a traves del tiempo (BPTT): la regla de
   la cadena recorre los pasos hacia atras, como las capas en la 1.3
3. Los gradientes se multiplican por Whh y por tanh' en cada paso: tienden
   a desaparecer (o explotar), y la red recuerda poco hacia atras
4. Recortar el gradiente (clip) evita que explote
5. Por caracteres no hay palabras desconocidas: la red aprende a deletrear

EXPERIMENTOS SUGERIDOS:
- Cambia H a 32 y a 256: como cambia la perdida y el texto generado?
- Muestrea con temperatura 0.5 y 1.5 (parametro de muestrear)
- Cambia LARGO a 5: que pasa con lo que la red aprende a recordar?
""")
