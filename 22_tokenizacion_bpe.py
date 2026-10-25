"""
LECCION 4.4 - TOKENIZACION BPE
==============================

Objetivo: entender que son los "tokens" de un modelo de lenguaje y como
se construye su vocabulario.

El mini-GPT de la 4.3 trabajaba con CARACTERES: vocabulario chico (61
simbolos), pero cada palabra ocupa muchas posiciones, asi que su ventana
de 64 posiciones abarcaba apenas unas 12 palabras. En la 3.1 se tokenizo
por PALABRAS: secuencias cortas, pero un vocabulario enorme y el problema
de las palabras nunca vistas.

BPE (Byte Pair Encoding, "codificacion por pares") busca el punto medio:
  1. empezar con los caracteres
  2. buscar el PAR de simbolos vecinos mas frecuente en el corpus
  3. fusionarlo en un simbolo nuevo ("q" + "u" -> "qu")
  4. repetir miles de veces
Las palabras frecuentes terminan siendo un solo token; las raras se
arman con pedazos. Es (con variantes) lo que usan GPT, Claude y casi
todos los modelos actuales.
"""

import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from corpus_es import cargar_quijote


# ============================================================
# PARTE 1: LOS DOS EXTREMOS
# ============================================================

texto = " ".join(cargar_quijote().lower().split())
corte = int(len(texto) * 0.95)
train, test = texto[:corte], texto[corte:]

palabras_train = re.findall(r"\w+|[^\w\s]", train)
palabras_test = re.findall(r"\w+|[^\w\s]", test)
vocab_palabras = set(palabras_train)
nuevas = [p for p in palabras_test if p not in vocab_palabras]
print("PARTE 1: dos formas de cortar")
print(f"  Caracteres: vocabulario {len(set(train))}, el test ocupa {len(test):,} posiciones")
print(f"  Palabras:   vocabulario {len(vocab_palabras):,}, el test ocupa {len(palabras_test):,} posiciones")
print(f"  Palabras del test que no estan en train: {len(nuevas)} "
      f"({len(nuevas) / len(palabras_test):.1%}), por ejemplo {sorted(set(nuevas))[:6]}")


# ============================================================
# PARTE 2: ENTRENAR BPE
# ============================================================
# Trabajamos palabra por palabra (los pares nunca cruzan de una palabra a
# otra). Cada palabra lleva delante el marcador "_" que representa el
# espacio, como hacen los tokenizadores reales: asi "_casa" (palabra
# completa) y "casa" (pedazo de "lacasa") son tokens distintos.
#
# Como "_que" aparece 20 000 veces, no la procesamos 20 000 veces: la
# guardamos una vez con su frecuencia.

def separar_palabras(t):
    return ["_" + p if p[0].isalnum() or p[0] == "_" else p
            for p in re.findall(r"\w+|[^\w\s]", t)]


frecuencias = Counter(separar_palabras(train))
partes = {p: list(p) for p in frecuencias}            # cada palabra, como lista de simbolos


def contar_pares():
    pares = Counter()
    for palabra, f in frecuencias.items():
        simbolos = partes[palabra]
        for a, b in zip(simbolos, simbolos[1:]):
            pares[a, b] += f
    return pares


def fusionar(a, b):
    nuevo = a + b
    for palabra in frecuencias:
        s = partes[palabra]
        if len(s) < 2 or a not in s:
            continue
        i, salida = 0, []
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                salida.append(nuevo)
                i += 2
            else:
                salida.append(s[i])
                i += 1
        partes[palabra] = salida


N_FUSIONES = 1000
fusiones = []
inicio = time.time()
for n in range(N_FUSIONES):
    pares = contar_pares()
    (a, b), f = pares.most_common(1)[0]
    fusionar(a, b)
    fusiones.append((a, b))
    if n < 12 or n in (99, 499, 999):
        print(f"  fusion {n + 1:4d}: {a!r:>8} + {b!r:<8} -> {a + b!r:<12} ({f:,} veces)")
print(f"Entrenamiento: {N_FUSIONES} fusiones en {time.time() - inicio:.0f} s")


# ============================================================
# PARTE 3: TOKENIZAR TEXTO NUEVO
# ============================================================
# Para tokenizar una palabra se aplican las fusiones EN EL MISMO ORDEN en
# que se aprendieron. rango[par] = en que turno se aprendio ese par.

rango = {par: i for i, par in enumerate(fusiones)}


def tokenizar_palabra(palabra, max_fusiones=N_FUSIONES):
    s = list(palabra)
    while len(s) > 1:
        candidatos = [(rango.get((s[i], s[i + 1]), 10 ** 9), i) for i in range(len(s) - 1)]
        r, i = min(candidatos)                         # el par aprendido mas temprano
        if r >= max_fusiones:
            break
        s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
    return s


def tokenizar(t, max_fusiones=N_FUSIONES):
    tokens = []
    for p in separar_palabras(t):
        tokens.extend(tokenizar_palabra(p, max_fusiones))
    return tokens


frase = "don quijote, el ingenioso hidalgo, conversaba con su computadora"
print("\nPARTE 3:", tokenizar(frase))
for palabra in ["_quijote", "_caballero", "_computadora", "_inteligencia", "_desfacer"]:
    print(f"  {palabra:<14} -> {tokenizar_palabra(palabra)}")


# ============================================================
# PARTE 4: CUANTO SE ACORTA EL TEXTO
# ============================================================
# Medimos cuantos tokens ocupa el test segun la cantidad de fusiones.
# Con 0 fusiones son caracteres (mas los marcadores "_").

muestra = test[:30000]
caracteres_muestra = len(muestra)
puntos = [0, 50, 100, 200, 500, 1000]
largos = []
for k in puntos:
    largos.append(len(tokenizar(muestra, k)))
    print(f"  {k:5d} fusiones: vocabulario ~{len(set(train)) + 1 + k:5d} | "
          f"{largos[-1]:6,} tokens | {caracteres_muestra / largos[-1]:.2f} caracteres por token")


# ============================================================
# PARTE 5: QUE TOKENS QUEDARON
# ============================================================
tokens_test = tokenizar(test)
uso = Counter(tokens_test)
print("\nPARTE 5: tokens mas usados en el test:", [t for t, _ in uso.most_common(15)])
largos_tokens = Counter(len(t.lstrip("_")) for t in set(a + b for a, b in fusiones))
print("  Largo (en letras) de los tokens aprendidos:", dict(sorted(largos_tokens.items())))


# ============================================================
# PARTE 6: GRAFICOS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(puntos, [caracteres_muestra / l for l in largos], "o-", color="#EC4899")
ax1.set_title("Compresion segun la cantidad de fusiones")
ax1.set_xlabel("Fusiones aprendidas")
ax1.set_ylabel("Caracteres por token")
ax1.grid(alpha=0.3)

frecs = sorted(uso.values(), reverse=True)
ax2.loglog(range(1, len(frecs) + 1), frecs, color="#EC4899")
ax2.set_title("Uso de cada token en el test (escala log-log)")
ax2.set_xlabel("Ranking del token")
ax2.set_ylabel("Apariciones")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("22_tokenizacion_bpe.png", dpi=120)
print("\nGrafico guardado en: 22_tokenizacion_bpe.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un token no es una palabra ni un caracter: es un pedazo de texto
   frecuente, aprendido de un corpus
2. BPE: empezar con caracteres y fusionar una y otra vez el par vecino
   mas frecuente
3. Para tokenizar, se aplican las fusiones en el orden aprendido
4. No hay palabras desconocidas: lo raro se arma con pedazos chicos
5. Mas fusiones = textos mas cortos (mas contexto en la misma ventana)
   pero un vocabulario mas grande (mas parametros en E y en la salida)

EXPERIMENTOS SUGERIDOS:
- Cambia N_FUSIONES a 3000: que palabras completas aparecen?
- Entrena BPE con las RESENAS de corpus_es.py y tokeniza el Quijote
- Tokeniza una frase en ingles: cuantos caracteres por token logra?
""")
