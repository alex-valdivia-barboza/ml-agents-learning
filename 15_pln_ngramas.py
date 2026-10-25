"""
LECCION 3.3 - MODELOS DE LENGUAJE CON N-GRAMAS
==============================================

Objetivo: el primer MODELO DE LENGUAJE: un programa que, dadas unas
palabras, da la probabilidad de cada palabra siguiente.

  P("panza" | "sancho") = ?

Es la misma pregunta que responde un modelo como Claude, solo que aqui
se responde CONTANDO: de todas las veces que aparece "sancho" en el
Quijote, que fraccion va seguida de "panza"?

Un n-grama es una secuencia de n tokens seguidos:
  unigrama: "sancho"     bigrama: "sancho panza"     trigrama: "dijo sancho panza"
Un modelo de n-gramas predice cada palabra mirando las n-1 anteriores.
"""

from collections import Counter, defaultdict
import math
import random
import re

import matplotlib.pyplot as plt

from corpus_es import cargar_quijote

random.seed(42)


# ============================================================
# PARTE 1: TOKENS CON PUNTUACION
# ============================================================
# A diferencia de la 3.1, aqui la puntuacion SI es un token: para
# generar texto hace falta saber donde van los puntos y las comas.

TOKEN = re.compile(r"[a-záéíóúüñ]+|[.,;:!?¿¡]")

texto = cargar_quijote().lower()
tokens = TOKEN.findall(texto)
corte = int(len(tokens) * 0.9)
train, test = tokens[:corte], tokens[corte:]
print(f"Tokens: {len(tokens):,} (train {len(train):,} | test {len(test):,})")


# ============================================================
# PARTE 2: CONTAR N-GRAMAS
# ============================================================
# conteos[contexto][palabra] = cuantas veces aparece "palabra" despues
# de "contexto". El contexto es una tupla con las n-1 palabras previas.
# Con n = 1 el contexto es la tupla vacia: solo frecuencias (como la 3.1).

def contar(tokens, n):
    conteos = defaultdict(Counter)
    for i in range(n - 1, len(tokens)):
        contexto = tuple(tokens[i - n + 1:i])
        conteos[contexto][tokens[i]] += 1
    return conteos


modelos = {n: contar(train, n) for n in (1, 2, 3)}
vocab = set(train)
V = len(vocab)
print(f"Vocabulario de train: {V:,} tokens")
for n in (1, 2, 3):
    print(f"  n={n}: {len(modelos[n]):,} contextos distintos, "
          f"{sum(len(c) for c in modelos[n].values()):,} n-gramas distintos")


# ============================================================
# PARTE 3: PROBABILIDAD DE LA SIGUIENTE PALABRA
# ============================================================
#   P(palabra | contexto) = conteo(contexto, palabra) / conteo(contexto)

def siguientes(contexto, n=2, k=5):
    conteo = modelos[n][tuple(contexto)]
    total = sum(conteo.values())
    return [(p, c / total) for p, c in conteo.most_common(k)], total


print("\n--- Que palabra sigue? (bigramas) ---")
for palabra in ["sancho", "don", "dijo", "señor", "."]:
    opciones, total = siguientes([palabra])
    print(f"despues de '{palabra}' ({total} veces):", ", ".join(f"{p} {q:.0%}" for p, q in opciones))
opciones, total = siguientes(["respondió", "sancho"], n=3)
print(f"despues de 'respondió sancho' ({total} veces):", ", ".join(f"{p} {q:.0%}" for p, q in opciones))


# ============================================================
# PARTE 4: GENERAR TEXTO
# ============================================================
# Partiendo de un contexto, se SORTEA la siguiente palabra con las
# probabilidades del modelo, se agrega, y se repite. random.choices
# elige un elemento con probabilidad proporcional a su peso.

def generar(inicio, n, largo=30):
    salida = list(inicio)
    for _ in range(largo):
        conteo = modelos[n].get(tuple(salida[len(salida) - n + 1:]) if n > 1 else ())
        if not conteo:                                   # contexto nunca visto: no sabe seguir
            break
        palabras, pesos = zip(*conteo.items())
        salida.append(random.choices(palabras, weights=pesos)[0])
    return " ".join(salida).replace(" ,", ",").replace(" .", ".")


print("\n--- Texto generado ---")
for n in (1, 2, 3):
    inicio = ["en", "un"][:max(n - 1, 0)] if n > 1 else []
    print(f"n={n}: {generar(inicio, n)}\n")


# ============================================================
# PARTE 5: EVALUAR CON PERPLEJIDAD
# ============================================================
# Un buen modelo le da probabilidad ALTA al texto real que no vio.
# Perplejidad = exp(-promedio de log P(palabra | contexto)) sobre test.
# Se interpreta como "entre cuantas palabras duda el modelo en promedio":
# mas baja es mejor.
#
# Problema: si un n-grama de test nunca aparecio en train, P = 0 y
# log(0) = -infinito. Solucion: SUAVIZADO aditivo (Laplace): sumar k a
# todos los conteos, para que nada tenga probabilidad cero.
#   P(w | ctx) = (conteo(ctx, w) + k) / (conteo(ctx) + k * V)

def probabilidad(contexto, palabra, n, k):
    conteo = modelos[n].get(tuple(contexto), Counter())
    return (conteo[palabra] + k) / (sum(conteo.values()) + k * V)


def perplejidad(tokens, n, k):
    log_suma, m = 0.0, 0
    for i in range(n - 1, len(tokens)):
        p = probabilidad(tokens[i - n + 1:i], tokens[i], n, k)
        log_suma += math.log(p)
        m += 1
    return math.exp(-log_suma / m)


muestra = test[:20000]
sin_ver = sum(1 for i in range(1, len(muestra)) if muestra[i] not in modelos[2].get((muestra[i - 1],), {}))
print(f"Bigramas de test que nunca aparecieron en train: {sin_ver / (len(muestra) - 1):.0%}")

print("\n--- Perplejidad en test (20 000 tokens) ---")
resultados = {}
for n in (1, 2, 3):
    for k in (1.0, 0.1, 0.01):
        resultados[(n, k)] = perplejidad(muestra, n, k)
        print(f"n={n} k={k:<5} | perplejidad = {resultados[(n, k)]:8.1f}")


# ============================================================
# PARTE 6: INTERPOLACION (mezclar 1, 2 y 3-gramas)
# ============================================================
# El suavizado de Laplace le quita demasiada probabilidad a lo que SI se
# vio. Mejor idea: combinar los tres modelos. Si el trigrama nunca se
# vio, su probabilidad es 0, pero el bigrama y el unigrama aportan.
#   P(w | a, b) = l1 * P1(w) + l2 * P2(w | b) + l3 * P3(w | a, b)
# con l1 + l2 + l3 = 1. Los pesos se ELIGEN con un tramo de validacion
# (leccion 1.6) y despues se mide una sola vez en test.

def p_relativa(contexto, palabra, n):
    conteo = modelos[n].get(tuple(contexto))
    return conteo[palabra] / sum(conteo.values()) if conteo else 0.0


def perplejidad_interpolada(tokens, lambdas):
    l1, l2, l3 = lambdas
    log_suma, m = 0.0, 0
    for i in range(2, len(tokens)):
        p = (l1 * probabilidad((), tokens[i], 1, 1.0)            # unigrama con Laplace: nunca es 0
             + l2 * p_relativa(tokens[i - 1:i], tokens[i], 2)
             + l3 * p_relativa(tokens[i - 2:i], tokens[i], 3))
        log_suma += math.log(p)
        m += 1
    return math.exp(-log_suma / m)


validacion = test[20000:40000]                       # otro tramo del 10% final, distinto de la muestra
candidatos = [(0.1, 0.3, 0.6), (0.2, 0.4, 0.4), (0.3, 0.5, 0.2), (0.4, 0.4, 0.2), (0.2, 0.6, 0.2)]
print("\n--- Interpolacion: elegir los pesos con validacion ---")
for lambdas in candidatos:
    print(f"l1, l2, l3 = {lambdas} | perplejidad en validacion = {perplejidad_interpolada(validacion, lambdas):.1f}")
mejores = min(candidatos, key=lambda l: perplejidad_interpolada(validacion, l))
pp_interpolada = perplejidad_interpolada(muestra, mejores)
print(f"Elegidos {mejores} -> perplejidad en TEST = {pp_interpolada:.1f}")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

opciones, _ = siguientes(["sancho"], k=10)
ax1.barh([p for p, _ in opciones][::-1], [q for _, q in opciones][::-1], color="#F59E0B")
ax1.set_title("P(palabra | 'sancho') - bigramas del Quijote")
ax1.set_xlabel("Probabilidad")

for k, color in zip((1.0, 0.1, 0.01), ("#9CA3AF", "#3B82F6", "#10B981")):
    ax2.plot([1, 2, 3], [resultados[(n, k)] for n in (1, 2, 3)], marker="o", color=color, label=f"k = {k}")
ax2.axhline(pp_interpolada, color="#F59E0B", linestyle="--", label="interpolada (1+2+3)")
ax2.set_yscale("log")
ax2.set_xticks([1, 2, 3])
ax2.set_title("Perplejidad en test (mas baja = mejor)")
ax2.set_xlabel("n (tamano del n-grama)")
ax2.set_ylabel("Perplejidad (log)")
ax2.legend()

plt.tight_layout()
plt.savefig("15_pln_ngramas.png", dpi=120)
print("\nGrafico guardado en: 15_pln_ngramas.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un modelo de lenguaje da P(siguiente palabra | palabras anteriores)
2. Los n-gramas lo estiman contando: conteo(contexto, w) / conteo(contexto)
3. Generar texto = sortear la siguiente palabra, agregarla y repetir
4. Con n mas grande el texto es mas coherente, pero la mayoria de los
   contextos largos nunca se vio: los datos no alcanzan
5. Perplejidad mide que tan sorprendido esta el modelo por texto nuevo;
   el suavizado evita probabilidades cero, y la interpolacion combina
   lo especifico (trigramas) con lo general (unigramas)

EXPERIMENTOS SUGERIDOS:
- Genera con n=4: el texto es mas coherente o copia frases enteras?
- Cambia generar() para elegir siempre la palabra mas probable (sin
  sortear): que pasa con el texto?
- Calcula la perplejidad del texto de train: por que es mucho mas baja?
""")
