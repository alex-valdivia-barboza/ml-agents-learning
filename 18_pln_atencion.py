"""
LECCION 3.6 - ATENCION
======================

Objetivo: el mecanismo central de los modelos de lenguaje actuales.

La RNN de la leccion 3.5 leia en orden y resumia todo en una memoria h.
Para usar algo de hace 40 pasos, el gradiente tenia que viajar 40 pasos
hacia atras y se desvanecia. La ATENCION propone otra cosa: que cada
posicion pueda MIRAR DIRECTAMENTE a todas las demas y decidir a cuales
prestar atencion, sin importar la distancia.

Se basa en una analogia con un diccionario:
  - CONSULTA (query): lo que estoy buscando
  - CLAVES (keys):    la etiqueta de cada elemento disponible
  - VALORES (values): el contenido de cada elemento
Un diccionario normal devuelve el valor cuya clave coincide EXACTAMENTE.
La atencion devuelve un PROMEDIO de todos los valores, pesado por cuanto
se parece cada clave a la consulta. Es un diccionario "blando", y por eso
se puede entrenar con gradientes.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)


# ============================================================
# PARTE 1: ATENCION CON NUMEROS A MANO
# ============================================================
# 3 elementos con claves y valores de 2 dimensiones, y una consulta.
# pesos = softmax(claves @ consulta)     salida = pesos @ valores

def softmax(z, eje=-1):
    z = z - z.max(axis=eje, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=eje, keepdims=True)


claves = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
valores = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
consulta = np.array([2.0, 0.0])                    # "busco algo parecido a la primera clave"
puntajes = claves @ consulta                       # que tanto coincide cada clave
pesos = softmax(puntajes)
salida = pesos @ valores
print("Puntajes:", puntajes, "-> pesos:", np.round(pesos, 3))
print("Salida (promedio pesado de los valores):", np.round(salida, 2))


# ============================================================
# PARTE 2: UNA TAREA QUE EXIGE "BUSCAR"
# ============================================================
# Cada ejemplo es una lista de 8 pares (clave, valor) y una consulta:
#     c:7  a:2  h:0  e:5  b:9  j:1  d:3  f:8   ?  e   -> respuesta: 5
# Claves: letras a..j. Valores: digitos 0..9. La respuesta es el valor
# cuya clave es igual a la consulta. Para acertar hay que encontrar el
# par correcto, este donde este.

N_CLAVES, N_VALORES, N_PARES = 10, 10, 8
LETRAS = "abcdefghij"


def generar(n):
    claves = np.array([rng.permutation(N_CLAVES)[:N_PARES] for _ in range(n)])   # (n, 8), sin repetir
    valores = rng.integers(0, N_VALORES, size=(n, N_PARES))                       # (n, 8)
    posicion = rng.integers(0, N_PARES, size=n)                                   # cual se consulta
    consultas = claves[np.arange(n), posicion]
    respuestas = valores[np.arange(n), posicion]
    return claves, valores, consultas, respuestas, posicion


train = generar(20000)
test = generar(2000)
k, v, q, r, _ = test
print("\nEjemplo:", "  ".join(f"{LETRAS[a]}:{b}" for a, b in zip(k[0], v[0])), f"  ?  {LETRAS[q[0]]}  -> {r[0]}")


# ============================================================
# PARTE 3: EL MODELO (una capa de atencion)
# ============================================================
# Cada clave y cada valor tiene un embedding de D numeros (leccion 3.4).
#   consulta_vec = Ek[q] @ Wq                       (D,)
#   claves_vec   = Ek[k_i] @ Wk                     (8, D)
#   puntajes     = claves_vec @ consulta_vec / raiz(D)
#   pesos        = softmax(puntajes)                <- la atencion
#   salida       = pesos @ Ev[v_i]                  (D,)
#   P(respuesta) = softmax(salida @ Wo)             (10,)

D = 16
P = {
    "Ek": rng.standard_normal((N_CLAVES, D)) * 0.5,
    "Ev": rng.standard_normal((N_VALORES, D)) * 0.5,
    "Wq": rng.standard_normal((D, D)) / np.sqrt(D),
    "Wk": rng.standard_normal((D, D)) / np.sqrt(D),
    "Wo": rng.standard_normal((D, N_VALORES)) / np.sqrt(D),
}


def forward(P, k, v, q):
    eq, ek, ev = P["Ek"][q], P["Ek"][k], P["Ev"][v]             # (B,D) (B,8,D) (B,8,D)
    cq, ck = eq @ P["Wq"], ek @ P["Wk"]                          # consulta y claves proyectadas
    puntajes = np.einsum("bd,bnd->bn", cq, ck) / np.sqrt(D)      # (B, 8)
    alfa = softmax(puntajes)                                     # (B, 8) pesos de atencion
    salida = np.einsum("bn,bnd->bd", alfa, ev)                   # (B, D)
    probs = softmax(salida @ P["Wo"])                            # (B, 10)
    cache = (eq, ek, ev, cq, ck, alfa, salida, probs)
    return probs, cache


def backward(P, k, v, q, r, cache):
    """La regla de la cadena de la leccion 1.3, paso a paso hacia atras."""
    eq, ek, ev, cq, ck, alfa, salida, probs = cache
    B = len(r)
    G = {n: np.zeros_like(m) for n, m in P.items()}
    dlogits = probs.copy()
    dlogits[np.arange(B), r] -= 1
    dlogits /= B                                                 # softmax + cross-entropy (1.4)
    G["Wo"] = salida.T @ dlogits
    dsalida = dlogits @ P["Wo"].T                                # (B, D)
    dalfa = np.einsum("bd,bnd->bn", dsalida, ev)                 # cuanto conviene cambiar cada peso
    dev = alfa[:, :, None] * dsalida[:, None, :]                 # (B, 8, D)
    dpunt = alfa * (dalfa - (alfa * dalfa).sum(axis=1, keepdims=True))   # derivada de softmax
    dpunt /= np.sqrt(D)
    dcq = np.einsum("bn,bnd->bd", dpunt, ck)
    dck = dpunt[:, :, None] * cq[:, None, :]
    G["Wq"] = eq.T @ dcq
    G["Wk"] = np.einsum("bnd,bne->de", ek, dck)
    np.add.at(G["Ek"], q, dcq @ P["Wq"].T)                       # la consulta y las claves
    np.add.at(G["Ek"], k, dck @ P["Wk"].T)                       # comparten embeddings
    np.add.at(G["Ev"], v, dev)
    return G


def perdida(P, k, v, q, r):
    probs, _ = forward(P, k, v, q)
    return -np.mean(np.log(probs[np.arange(len(r)), r] + 1e-12))


# Gradient checking (leccion 1.3) sobre 3 numeros de cada parametro
kb, vb, qb, rb, _ = generar(16)
_, cache = forward(P, kb, vb, qb)
G = backward(P, kb, vb, qb, rb, cache)
peor = 0.0
for nombre in P:
    for _ in range(3):
        idx = tuple(rng.integers(0, s) for s in P[nombre].shape)
        original = P[nombre][idx]
        P[nombre][idx] = original + 1e-5
        mas = perdida(P, kb, vb, qb, rb)
        P[nombre][idx] = original - 1e-5
        menos = perdida(P, kb, vb, qb, rb)
        P[nombre][idx] = original
        numerico = (mas - menos) / 2e-5
        peor = max(peor, abs(numerico - G[nombre][idx]) / (abs(numerico) + abs(G[nombre][idx]) + 1e-12))
print(f"\nGradient checking: peor error relativo = {peor:.1e}")


# ============================================================
# PARTE 4: ENTRENAMIENTO
# ============================================================

def exactitud(P, datos):
    k, v, q, r, _ = datos
    return np.mean(forward(P, k, v, q)[0].argmax(axis=1) == r)


TASA, EPOCAS, LOTE = 0.5, 3, 100
historial = []                                     # (mini-batches vistos, perdida test, exactitud test)
k, v, q, r, _ = train
pasos = 0
for epoca in range(EPOCAS):
    orden = rng.permutation(len(r))
    for i in range(0, len(r), LOTE):
        if pasos % 10 == 0:                        # medir cada 10 mini-batches
            historial.append((pasos, perdida(P, *test[:4]), exactitud(P, test)))
        lote = orden[i:i + LOTE]
        _, cache = forward(P, k[lote], v[lote], q[lote])
        G = backward(P, k[lote], v[lote], q[lote], r[lote], cache)
        for n in P:
            P[n] -= TASA * G[n]
        pasos += 1
    print(f"Epoca {epoca} | perdida test = {perdida(P, *test[:4]):.3f} | exactitud test = {exactitud(P, test):.1%}")
for pasos_vistos, _, exac in historial[:12]:
    print(f"  despues de {pasos_vistos:3d} mini-batches: exactitud test = {exac:.1%}")


# ============================================================
# PARTE 5: A DONDE MIRA EL MODELO
# ============================================================

k, v, q, r, pos = test
probs, cache = forward(P, k, v, q)
alfa = cache[5]
print(f"\nPeso de atencion promedio en el par correcto: {alfa[np.arange(len(r)), pos].mean():.1%}")
for p in range(N_PARES):
    m = pos == p
    print(f"  par correcto en la posicion {p}: exactitud {np.mean(probs[m].argmax(axis=1) == r[m]):.1%}")
print("\nEjemplo:", "  ".join(f"{LETRAS[a]}:{b}" for a, b in zip(k[0], v[0])), f" ? {LETRAS[q[0]]}")
print("Atencion:", "  ".join(f"{a:.2f}" for a in alfa[0]), f"-> responde {probs[0].argmax()} (correcto: {r[0]})")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot([h[0] for h in historial], [h[2] for h in historial], color="#10B981", marker="o", markersize=3)
ax1.axhline(1 / N_VALORES, color="gray", linestyle="--", label="adivinar (10%)")
ax1.set_title("Exactitud en test durante el entrenamiento")
ax1.set_xlabel("Mini-batches")
ax1.legend()

ax2.imshow(alfa[:12], cmap="Oranges", vmin=0, vmax=1)
for fila in range(12):
    ax2.add_patch(plt.Rectangle((pos[fila] - 0.5, fila - 0.5), 1, 1, fill=False, edgecolor="#3B82F6", linewidth=2))
    for col in range(N_PARES):
        ax2.text(col, fila, f"{LETRAS[k[fila, col]]}:{v[fila, col]}", ha="center", va="center", fontsize=7)
ax2.set_yticks(range(12))
ax2.set_yticklabels([f"? {LETRAS[x]}" for x in q[:12]])
ax2.set_xticks(range(N_PARES))
ax2.set_title("Pesos de atencion (recuadro azul = par correcto)")
ax2.set_xlabel("Posicion del par")

plt.tight_layout()
plt.savefig("18_pln_atencion.png", dpi=120)
print("\nGrafico guardado en: 18_pln_atencion.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Atencion = diccionario blando: pesos = softmax(consulta . claves),
   salida = promedio de los valores pesado por esos pesos
2. Consulta, claves y valores salen de embeddings multiplicados por
   matrices aprendidas (Wq, Wk)
3. Se divide por raiz(D) para que los puntajes no saturen la softmax
4. Cada posicion accede directo a cualquier otra: la distancia no importa
   (a diferencia de la RNN de la 3.5)
5. Todo se entrena con la misma regla de la cadena, verificada con
   gradient checking

EXPERIMENTOS SUGERIDOS:
- Cambia N_PARES a 20: sigue funcionando? cuanto tarda?
- Quita la division por np.sqrt(D) en forward y backward
- Reemplaza la atencion por el promedio simple de los valores (alfa
  igual a 1/8 para todos): que exactitud se puede lograr?
""")
