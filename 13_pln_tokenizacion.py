"""
LECCION 3.1 - DEL TEXTO A LOS NUMEROS
=====================================

Objetivo: convertir texto en algo que un programa pueda procesar.

Todas las redes del modulo 1 recibian numeros: coordenadas (x1, x2) o
pixeles. Un texto es una cadena de caracteres. Antes de cualquier modelo
de lenguaje hay que decidir:
  1. NORMALIZAR: que diferencias ignorar (mayusculas, tildes...)
  2. TOKENIZAR: en que unidades partir el texto (palabras, aqui)
  3. VOCABULARIO: que tokens conocemos y que numero le toca a cada uno

Usamos "Don Quijote de la Mancha" (unas 380 000 palabras). La primera
vez se descarga a data/.
"""

from collections import Counter
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np

from corpus_es import cargar_quijote


# ============================================================
# PARTE 1: EL TEXTO CRUDO
# ============================================================

texto = cargar_quijote()
print(f"Caracteres: {len(texto):,}")
comienzo = texto.index("En un lugar de la Mancha")       # el comienzo de la novela
print(f"Los 300 caracteres del comienzo:\n{texto[comienzo:comienzo + 300]}\n")


# ============================================================
# PARTE 2: NORMALIZAR
# ============================================================
# "En", "en" y "EN" son la misma palabra para casi cualquier tarea.
# Pasar todo a minusculas reduce el vocabulario sin perder casi nada.
# Quitar tildes es una decision mas discutible: une "esta" y "esta",
# pero tambien "el" (articulo) y "el" (pronombre). Lo hacemos opcional.

def quitar_tildes(texto):
    # NFD separa cada letra de su tilde ("a" + "´"); despues se descartan las marcas
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in descompuesto if unicodedata.category(ch) != "Mn")


def normalizar(texto, sin_tildes=False):
    texto = texto.lower()
    return quitar_tildes(texto) if sin_tildes else texto


# ============================================================
# PARTE 3: TOKENIZAR
# ============================================================
# Opcion ingenua: separar por espacios. Problema: la puntuacion queda
# pegada ("mancha," y "mancha" serian palabras distintas).
# Opcion mejor: una expresion regular que tome secuencias de letras.

ejemplo = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo..."
print("split():  ", normalizar(ejemplo).split())
TOKEN = re.compile(r"[a-záéíóúüñ]+")
print("regex:    ", TOKEN.findall(normalizar(ejemplo)))


def tokenizar(texto, sin_tildes=False):
    return TOKEN.findall(normalizar(texto, sin_tildes))


tokens = tokenizar(texto)
print(f"\nTokens (palabras) en el Quijote: {len(tokens):,}")


# ============================================================
# PARTE 4: VOCABULARIO
# ============================================================
# Counter cuenta cuantas veces aparece cada token. El vocabulario asigna
# un numero entero a cada palabra distinta: ese numero es lo que vera
# un modelo. Las palabras raras se agrupan en un token especial <UNK>
# (desconocida) para no tener un vocabulario enorme.

frecuencias = Counter(tokens)
print(f"Palabras distintas: {len(frecuencias):,}")
print("Las 10 mas frecuentes:", frecuencias.most_common(10))


def construir_vocabulario(frecuencias, minimo=1):
    vocab = {"<UNK>": 0}
    for palabra, n in frecuencias.most_common():
        if n >= minimo:
            vocab[palabra] = len(vocab)
    return vocab


def codificar(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]


def decodificar(indices, vocab):
    inverso = {i: p for p, i in vocab.items()}
    return [inverso[i] for i in indices]


vocab = construir_vocabulario(frecuencias, minimo=5)
frase = tokenizar("Sancho Panza era un labrador vecino suyo, hombre de bien")
codigos = codificar(frase, vocab)
print(f"\nVocabulario con palabras que aparecen 5 o mas veces: {len(vocab):,} entradas")
print(f"Frase:        {frase}")
print(f"Codificada:   {codigos}")
print(f"Decodificada: {decodificar(codigos, vocab)}")

sin_tildes = Counter(tokenizar(texto, sin_tildes=True))
print(f"\nPalabras distintas con tildes: {len(frecuencias):,} | sin tildes: {len(sin_tildes):,}")


# ============================================================
# PARTE 5: LA LEY DE ZIPF
# ============================================================
# Si se ordenan las palabras por frecuencia, la frecuencia de la palabra
# en el puesto r es aproximadamente proporcional a 1/r. Pocas palabras
# aparecen muchisimo y muchisimas aparecen muy poco.

conteos = np.array(sorted(frecuencias.values(), reverse=True))
rangos = np.arange(1, len(conteos) + 1)
pendiente = np.polyfit(np.log(rangos[:1000]), np.log(conteos[:1000]), 1)[0]
una_vez = np.sum(conteos == 1)
print(f"\nPendiente en escala log-log (primeras 1000): {pendiente:.2f}  (Zipf ideal: -1)")
print(f"Palabras que aparecen una sola vez: {una_vez:,} ({una_vez / len(conteos):.0%} del vocabulario)")
print(f"Las 100 palabras mas frecuentes cubren el {conteos[:100].sum() / conteos.sum():.0%} del texto")


# ============================================================
# PARTE 6: PALABRAS DESCONOCIDAS EN TEXTO NUEVO
# ============================================================
# Construimos el vocabulario con el primer 80% del libro y medimos cuantas
# palabras del 20% final no conoce: es lo que le pasa a un modelo con
# texto que no vio al entrenar (como el test de la leccion 1.4).

corte = int(len(tokens) * 0.8)
frec_train = Counter(tokens[:corte])
prueba = tokens[corte:]
print("\n--- Cobertura del vocabulario en el 20% final del libro ---")
cobertura = []
for minimo in [1, 2, 5, 10, 50]:
    v = construir_vocabulario(frec_train, minimo)
    desconocidas = sum(1 for t in prueba if t not in v)
    cobertura.append((len(v), 1 - desconocidas / len(prueba)))
    print(f"minimo={minimo:<3} | vocabulario={len(v):6,} | palabras desconocidas: {desconocidas / len(prueba):.1%}")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

ax1.loglog(rangos, conteos, color="#3B82F6", label="Quijote")
ax1.loglog(rangos, conteos[0] / rangos, color="gray", linestyle="--", label="Zipf: f = f1 / r")
ax1.set_title("Ley de Zipf")
ax1.set_xlabel("Rango de la palabra (1 = la mas frecuente)")
ax1.set_ylabel("Frecuencia")
ax1.legend()

top = frecuencias.most_common(20)
ax2.barh([p for p, _ in top][::-1], [n for _, n in top][::-1], color="#F59E0B")
ax2.set_title("Las 20 palabras mas frecuentes")
ax2.set_xlabel("Apariciones")

ax3.plot([v for v, _ in cobertura], [c * 100 for _, c in cobertura], marker="o", color="#10B981")
ax3.set_xscale("log")
ax3.set_title("Cobertura del texto nuevo segun el vocabulario")
ax3.set_xlabel("Tamano del vocabulario (log)")
ax3.set_ylabel("% de palabras conocidas")

plt.tight_layout()
plt.savefig("13_pln_tokenizacion.png", dpi=120)
print("\nGrafico guardado en: 13_pln_tokenizacion.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Normalizar decide que diferencias ignorar (mayusculas, tildes)
2. Tokenizar parte el texto en unidades; la puntuacion complica el split
3. Un vocabulario asigna un numero entero a cada token: eso ve el modelo
4. Ley de Zipf: pocas palabras son muy frecuentes y la mayoria son raras
5. Siempre habra palabras desconocidas en texto nuevo: hace falta <UNK>
   (o tokens mas chicos que la palabra, como en la leccion 4.4)

EXPERIMENTOS SUGERIDOS:
- Tokeniza con sin_tildes=True: que palabras distintas se fusionan?
- Cambia la regex para conservar los numeros: cuantos tokens nuevos hay?
- Calcula la ley de Zipf con las RESENAS de corpus_es.py: se cumple
  con tan poco texto?
""")
