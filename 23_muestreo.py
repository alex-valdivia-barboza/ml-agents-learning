"""
LECCION 4.5 - GENERAR TEXTO: MUESTREO
=====================================

Objetivo: entender como se elige cada token a partir de las
probabilidades que da el modelo, y por que esa eleccion cambia tanto el
texto que sale.

Un modelo de lenguaje no escribe texto: da una DISTRIBUCION de
probabilidad para el siguiente token. Quien genera tiene que decidir que
hacer con ella. Las opciones clasicas:
  - GREEDY: elegir siempre el mas probable
  - MUESTREO: sortear segun las probabilidades
  - TEMPERATURA: estirar o achatar la distribucion antes de sortear
  - TOP-K: sortear solo entre los k mas probables
  - TOP-P (nucleus): sortear entre los mas probables que suman p
Son los mismos controles que ofrecen las API de los modelos grandes.

Usa el mini-GPT entrenado en la leccion 4.3 (data/mini_gpt.npz):
ejecuta 21_mini_gpt.py antes que este script.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

import mini_transformer as mt
from corpus_es import cargar_quijote

rng = np.random.default_rng(42)

if not os.path.exists("data/mini_gpt.npz"):
    raise SystemExit("Falta data/mini_gpt.npz: ejecuta primero 21_mini_gpt.py (leccion 4.3)")
modelo, archivo = mt.cargar("data/mini_gpt.npz")
caracteres = [str(c) for c in archivo["caracteres"]]
a_indice = {ch: i for i, ch in enumerate(caracteres)}
T = modelo["P"].shape[0]


def logits_siguiente(texto):
    tokens = np.array([[a_indice[ch] for ch in texto[-T:]]])
    return mt.forward(modelo, tokens).data[0, -1]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ============================================================
# PARTE 1: LO QUE DA EL MODELO
# ============================================================
contexto = "dijo don quijote a su escud"
probs = softmax(logits_siguiente(contexto))
orden = np.argsort(probs)[::-1]
print(f"PARTE 1: despues de {contexto!r}, los 8 caracteres mas probables:")
for i in orden[:8]:
    print(f"   {caracteres[i]!r}: {probs[i]:.3f}")


# ============================================================
# PARTE 2: CINCO FORMAS DE ELEGIR
# ============================================================
# Todas reciben los logits y devuelven el indice del caracter elegido.

def greedy(logits):
    return int(np.argmax(logits))


def con_temperatura(logits, temperatura=1.0):
    return int(rng.choice(len(logits), p=softmax(logits / temperatura)))


def top_k(logits, k=5, temperatura=1.0):
    corte = np.sort(logits)[-k]                       # el k-esimo puntaje mas alto
    filtrados = np.where(logits >= corte, logits, -np.inf)
    return con_temperatura(filtrados, temperatura)


def top_p(logits, p=0.9, temperatura=1.0):
    probs = softmax(logits / temperatura)
    orden = np.argsort(probs)[::-1]
    acumulada = np.cumsum(probs[orden])
    n = np.searchsorted(acumulada, p) + 1             # cuantos hacen falta para llegar a p
    elegidos = orden[:n]
    return int(rng.choice(elegidos, p=probs[elegidos] / probs[elegidos].sum()))


def generar(inicio, n, elegir):
    texto = inicio
    for _ in range(n):
        texto += caracteres[elegir(logits_siguiente(texto))]
    return texto


estrategias = {
    "greedy": greedy,
    "T = 0.5": lambda z: con_temperatura(z, 0.5),
    "T = 1.0": lambda z: con_temperatura(z, 1.0),
    "T = 1.5": lambda z: con_temperatura(z, 1.5),
    "top-k (k = 5)": lambda z: top_k(z, 5),
    "top-p (p = 0.9)": lambda z: top_p(z, 0.9),
}
print("\nPARTE 2: el mismo inicio con cada estrategia")
for nombre, elegir in estrategias.items():
    print(f"\n[{nombre}]\n  {generar('sancho dijo: ', 150, elegir)}")


# ============================================================
# PARTE 3: COMO CAMBIA LA DISTRIBUCION
# ============================================================
logits = logits_siguiente(contexto)
print(f"\nPARTE 3: probabilidad del mas probable y cuantos caracteres juntan el 90%")
for temperatura in [0.5, 1.0, 1.5]:
    pr = softmax(logits / temperatura)
    n90 = np.searchsorted(np.cumsum(np.sort(pr)[::-1]), 0.9) + 1
    print(f"  T = {temperatura}: maximo {pr.max():.2f} | {n90} caracteres juntan el 90%")


# ============================================================
# PARTE 4: MEDIR CALIDAD Y VARIEDAD
# ============================================================
# Para cada estrategia generamos 8 textos de 300 caracteres y medimos:
#   - palabras reales: % de las palabras generadas que existen en el Quijote
#   - variedad: % de trigramas de palabras distintos (1 = nunca se repite)

vocabulario = set(re.findall(r"\w+", cargar_quijote().lower()))
inicios = ["el caballero ", "sancho dijo: ", "la noche ", "y en esto ", "don quijote ",
           "la sobrina ", "el cura ", "dulcinea "]
resultados = {}
for nombre, elegir in estrategias.items():
    palabras = []
    trigramas, total_trigramas = set(), 0
    for inicio in inicios:
        texto = generar(inicio, 300, elegir)[len(inicio):]
        p = re.findall(r"\w+", texto)[1:-1]          # sin la primera y la ultima (pueden quedar cortadas)
        palabras.extend(p)
        tri = list(zip(p, p[1:], p[2:]))
        trigramas.update(tri)
        total_trigramas += len(tri)
    reales = np.mean([w in vocabulario for w in palabras])
    variedad = len(trigramas) / total_trigramas
    resultados[nombre] = (reales, variedad)
print("\nPARTE 4: calidad y variedad")
for nombre, (reales, variedad) in resultados.items():
    print(f"  {nombre:<16} palabras reales {reales:.0%} | trigramas distintos {variedad:.0%}")


# ============================================================
# PARTE 5: GRAFICOS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(12)
for temperatura, color in [(0.5, "#3B82F6"), (1.0, "#EC4899"), (1.5, "#F59E0B")]:
    pr = softmax(logits / temperatura)[orden[:12]]
    ax1.plot(x, pr, "o-", color=color, label=f"T = {temperatura}")
ax1.set_xticks(x)
ax1.set_xticklabels([repr(caracteres[i]) for i in orden[:12]])
ax1.set_title(f"Siguiente caracter despues de {contexto!r}")
ax1.set_ylabel("Probabilidad")
ax1.legend()
ax1.grid(alpha=0.3)

for nombre, (reales, variedad) in resultados.items():
    ax2.scatter(variedad * 100, reales * 100, s=60, color="#EC4899")
    ax2.annotate(nombre, (variedad * 100, reales * 100), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax2.set_xlabel("Trigramas distintos (%)  -> mas variedad")
ax2.set_ylabel("Palabras reales (%)  -> mas correcto")
ax2.set_title("Calidad contra variedad")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("23_muestreo.png", dpi=120)
print("\nGrafico guardado en: 23_muestreo.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. El modelo da probabilidades; la estrategia de muestreo decide el texto
2. Greedy es determinista y cae en bucles repetitivos
3. La temperatura divide los logits: T < 1 concentra, T > 1 reparte
4. Top-k y top-p cortan la "cola" de opciones improbables, que es donde
   salen los errores, sin perder variedad entre las opciones buenas
5. Hay un compromiso entre texto correcto y texto variado

EXPERIMENTOS SUGERIDOS:
- Prueba top_p con p = 0.5 y con p = 0.99
- Combina: top_k(z, 10, temperatura=1.3)
- Cambia la semilla de rng y vuelve a generar con greedy y con T = 1.0
""")
