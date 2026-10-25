"""
LECCION 3.4 - EMBEDDINGS DE PALABRAS (WORD2VEC)
===============================================

Objetivo: representar cada palabra con un vector de numeros REALES, de
forma que palabras con significado parecido tengan vectores parecidos.

En las lecciones anteriores cada palabra era un simbolo aislado (un
indice, un one-hot): "buena" y "buenos" no tenian nada en comun. La idea
de los embeddings viene de una frase famosa de la linguistica:

    "Conoceras una palabra por la compania que mantiene" (J. R. Firth)

Palabras que aparecen rodeadas de las mismas palabras suelen significar
cosas parecidas. Word2vec (2013) aprende vectores con esa idea usando
una sola tarea: dada una palabra, adivinar que palabras aparecen cerca.
No hacen falta etiquetas: el propio texto es el ejemplo.
"""

from collections import Counter
import re

import matplotlib.pyplot as plt
import numpy as np

from corpus_es import cargar_quijote

rng = np.random.default_rng(42)


# ============================================================
# PARTE 1: VOCABULARIO Y SUBMUESTREO
# ============================================================
# Solo palabras que aparecen al menos 10 veces (las raras no tienen
# suficientes contextos para aprender nada). Ademas, las palabras muy
# frecuentes ("que", "de") se descartan al azar con cierta probabilidad:
# aparecen en todos lados y aportan poco sobre sus vecinas.

tokens = re.findall(r"[a-záéíóúüñ]+", cargar_quijote().lower())
conteo = Counter(tokens)
palabras = [p for p, n in conteo.most_common() if n >= 10]
indice = {p: i for i, p in enumerate(palabras)}
V = len(palabras)

frecuencia = np.array([conteo[p] for p in palabras]) / len(tokens)
prob_conservar = np.minimum(1.0, np.sqrt(1e-3 / frecuencia) + 1e-3 / frecuencia)
ids = np.array([indice[t] for t in tokens if t in indice])
ids = ids[rng.random(len(ids)) < prob_conservar[ids]]
print(f"Vocabulario: {V:,} palabras | tokens tras submuestreo: {len(ids):,} de {len(tokens):,}")
print(f"P(conservar 'que') = {prob_conservar[indice['que']]:.2f} | P(conservar 'sancho') = {prob_conservar[indice['sancho']]:.2f}")


# ============================================================
# PARTE 2: PARES (PALABRA, VECINA)
# ============================================================
# Para cada posicion, las palabras a distancia 1 o 2 son sus "vecinas".
# Cada par (centro, vecina) es un ejemplo POSITIVO: "estas dos palabras
# aparecieron juntas".

VENTANA = 2
centros, vecinas = [], []
for d in range(1, VENTANA + 1):
    centros += [ids[:-d], ids[d:]]
    vecinas += [ids[d:], ids[:-d]]
centros, vecinas = np.concatenate(centros), np.concatenate(vecinas)
print(f"Pares (centro, vecina): {len(centros):,}")
print("Ejemplos:", [(palabras[c], palabras[v]) for c, v in zip(centros[100:105], vecinas[100:105])])


# ============================================================
# PARTE 3: EL MODELO (skip-gram con muestreo negativo)
# ============================================================
# Cada palabra tiene DOS vectores de dimension D:
#   E[i] = su embedding cuando es la palabra central
#   C[i] = su vector cuando es vecina (contexto)
# Para un par (c, v): z = E[c] . C[v]   y   P(son vecinas) = sigmoid(z)
# Es la neurona de la leccion 1.1 haciendo clasificacion binaria:
#   - pares reales             -> etiqueta 1
#   - pares inventados al azar -> etiqueta 0 (muestras NEGATIVAS)
# Las negativas se sortean con probabilidad proporcional a frecuencia^0.75.

D = 50
NEGATIVAS = 5
E = (rng.random((V, D)) - 0.5) / D
C = np.zeros((V, D))
dist_neg = frecuencia ** 0.75
dist_neg /= dist_neg.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def paso(c, v, tasa):
    """Un mini-batch: B pares positivos y B*NEGATIVAS negativos."""
    B = len(c)
    neg = rng.choice(V, size=(B, NEGATIVAS), p=dist_neg)
    ctx = np.concatenate([v[:, None], neg], axis=1)              # (B, 1+K): primero la real
    etiquetas = np.zeros((B, 1 + NEGATIVAS))
    etiquetas[:, 0] = 1.0
    e = E[c]                                                     # (B, D)
    vecs = C[ctx]                                                # (B, 1+K, D)
    z = np.einsum("bd,bkd->bk", e, vecs)                         # producto punto de cada par
    p = sigmoid(z)
    perdida = -np.mean(np.log(np.where(etiquetas == 1, p, 1 - p) + 1e-10))
    g = p - etiquetas                                            # gradiente de cross-entropy (1.4)
    grad_e = np.einsum("bk,bkd->bd", g, vecs)                    # (B, D)
    grad_c = g[:, :, None] * e[:, None, :]                       # (B, 1+K, D)
    np.add.at(E, c, -tasa * grad_e)                              # acumula si una palabra se repite
    np.add.at(C, ctx, -tasa * grad_c)
    return perdida


# ============================================================
# PARTE 4: ENTRENAMIENTO
# ============================================================

EPOCAS, B = 5, 512
historial = []
total = EPOCAS * (len(centros) // B)
paso_actual = 0
for epoca in range(EPOCAS):
    orden = rng.permutation(len(centros))
    perdidas = []
    for i in range(0, len(orden) - B + 1, B):
        tasa = 0.05 * (1 - paso_actual / total) + 0.001             # la tasa baja de a poco
        lote = orden[i:i + B]
        perdidas.append(paso(centros[lote], vecinas[lote], tasa))
        paso_actual += 1
    historial += perdidas
    print(f"Epoca {epoca + 1}: perdida media = {np.mean(perdidas):.4f}")


# ============================================================
# PARTE 5: QUE APRENDIO: VECINOS MAS CERCANOS
# ============================================================
# Similitud coseno: el coseno del angulo entre dos vectores.
#   1 = misma direccion, 0 = sin relacion, -1 = opuestos.
# Antes se resta el vector promedio: todos los embeddings comparten una
# direccion comun (la de "ser una palabra frecuente del libro") que haria
# parecer similares a todos entre si.

centrados = E - E.mean(axis=0)
normalizados = centrados / np.linalg.norm(centrados, axis=1, keepdims=True)


def similares(palabra, k=6):
    sims = normalizados @ normalizados[indice[palabra]]
    mejores = np.argsort(-sims)[1:k + 1]
    return [(palabras[j], sims[j]) for j in mejores]


print("\n--- Palabras mas parecidas (similitud coseno) ---")
for p in ["sancho", "caballero", "dulcinea", "dijo", "casa", "buena", "noche", "oro"]:
    print(f"{p:10s} -> " + ", ".join(f"{q} ({s:.2f})" for q, s in similares(p)))


# ============================================================
# PARTE 6: VISUALIZACION (proyeccion a 2D con PCA)
# ============================================================
# Los vectores tienen 50 dimensiones. PCA busca las 2 direcciones en las
# que los puntos varian mas y los proyecta ahi, para poder dibujarlos.

grupos = {
    "personajes": ["sancho", "quijote", "dulcinea", "rocinante", "cura", "barbero", "duque", "duquesa"],
    "tiempo": ["noche", "día", "mañana", "tarde", "hora", "año"],
    "decir": ["dijo", "respondió", "replicó", "preguntó", "dice", "decía"],
    "números": ["dos", "tres", "cuatro", "cinco", "seis", "cien"],
}
seleccion = [p for g in grupos.values() for p in g if p in indice]
M = normalizados[[indice[p] for p in seleccion]]
M = M - M.mean(axis=0)
_, _, Vt = np.linalg.svd(M, full_matrices=False)
puntos = M @ Vt[:2].T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})
colores = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
i = 0
for (nombre, lista), color in zip(grupos.items(), colores):
    for p in lista:
        if p in indice:
            ax1.scatter(*puntos[i], color=color)
            ax1.annotate(p, puntos[i], fontsize=9, xytext=(3, 3), textcoords="offset points")
            i += 1
    ax1.scatter([], [], color=color, label=nombre)
ax1.legend()
ax1.set_title("Embeddings proyectados a 2D (PCA)")

ventana = 200
ax2.plot(np.convolve(historial, np.ones(ventana) / ventana, mode="valid"), color="#F59E0B")
ax2.set_title("Perdida durante el entrenamiento (media movil)")
ax2.set_xlabel("Mini-batch")
ax2.set_ylabel("Cross-entropy")

plt.tight_layout()
plt.savefig("16_pln_embeddings.png", dpi=120)
print("\nGrafico guardado en: 16_pln_embeddings.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un embedding es un vector denso que representa una palabra; palabras
   usadas en contextos parecidos terminan con vectores parecidos
2. Word2vec lo aprende con una tarea sin etiquetas: predecir que palabras
   aparecen cerca (skip-gram)
3. Muestreo negativo: en vez de una softmax sobre todo el vocabulario,
   distinguir pares reales de pares al azar (clasificacion binaria)
4. La similitud coseno mide que tan parecidos son dos vectores
5. Los embeddings se aprenden con texto sin etiquetar y despues sirven
   para otras tareas (como clasificar resenas)

EXPERIMENTOS SUGERIDOS:
- Cambia VENTANA a 1 y a 5: los vecinos son mas "gramaticales" o mas
  "tematicos"?
- Usa D = 10 y D = 200: como cambian los vecinos?
- Busca los vecinos de "bueno", "malo", "hermosa": sirven para las
  resenas de la leccion 3.2?
""")
