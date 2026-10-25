"""
LECCION 4.2 - EL BLOQUE TRANSFORMER
===================================

Objetivo: armar, pieza por pieza, el bloque que se repite dentro de GPT,
de Claude y de todos los modelos de lenguaje actuales.

La capa de atencion de la 3.6 tenia una consulta y un conjunto de pares
clave-valor. Un bloque transformer generaliza eso:
  - TODAS las posiciones hacen de consulta, clave y valor a la vez
    (auto-atencion), y cada una solo puede mirar hacia ATRAS (mascara
    causal): asi el modelo sirve para predecir el siguiente token
  - varias CABEZAS de atencion trabajan en paralelo, cada una buscando
    algo distinto
  - un embedding de POSICION le dice al modelo donde esta cada token
  - NORMALIZACION y conexiones RESIDUALES permiten apilar muchos bloques

Las piezas estan en mini_transformer.py y solo tienen forward: los
gradientes los calcula mini_autograd (leccion 4.1).
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import mini_transformer as mt
from mini_autograd import Adam, Tensor, cross_entropy, parametros_en_cero

rng = np.random.default_rng(42)
np.set_printoptions(precision=2, suppress=True)


# ============================================================
# PARTE 1: LA MASCARA CAUSAL
# ============================================================
# Para predecir el siguiente token, la posicion t solo puede usar los
# tokens 0..t. Si pudiera mirar t+1, "predeciria" copiando la respuesta.
# La mascara suma -1e9 a los puntajes prohibidos: tras la softmax, valen 0.

T = 5
puntajes = rng.standard_normal((T, T))
mascara = mt.mascara_causal(T)
pesos = Tensor(puntajes + mascara).softmax().data
print("PARTE 1: pesos de atencion con mascara causal (fila = quien mira)")
print(pesos)
print("Cada fila suma:", pesos.sum(axis=1))


# ============================================================
# PARTE 2: VARIAS CABEZAS, Y SUS SHAPES
# ============================================================
# Con D = 8 y H = 2 cabezas, cada cabeza trabaja con vectores de 4
# numeros. Se calculan todas a la vez con arrays de 4 dimensiones.

B, T, D, H = 2, 5, 8, 2
bloque_chico = mt.init_bloque(rng, D)
x = Tensor(rng.standard_normal((B, T, D)))
q = (x @ bloque_chico["Wq"]).reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
alfas = []
salida = mt.atencion(x, bloque_chico, H, guardar=alfas)
print(f"\nPARTE 2: x {x.shape} -> consultas por cabeza {q.shape} "
      f"-> pesos {alfas[0].shape} -> salida {salida.shape}")


# ============================================================
# PARTE 3: COMPROBAR QUE NO MIRA EL FUTURO
# ============================================================
# Cambiamos los vectores de las posiciones 3 y 4. Las salidas de las
# posiciones 0, 1 y 2 tienen que quedar EXACTAMENTE iguales.

x2 = x.data.copy()
x2[:, 3:] = rng.standard_normal((B, 2, D))
salida2 = mt.bloque(Tensor(x2), bloque_chico, H)
salida1 = mt.bloque(x, bloque_chico, H)
cambio = np.abs(salida2.data - salida1.data).max(axis=(0, 2))
print("\nPARTE 3: cuanto cambia la salida de cada posicion al cambiar las posiciones 3 y 4")
print("  ", cambio)


# ============================================================
# PARTE 4: SIN POSICIONES, EL ORDEN NO EXISTE
# ============================================================
# La atencion compara vectores: si se desordenan las entradas, las salidas
# se desordenan igual, pero no cambian. Sin mascara y sin embeddings de
# posicion, "el perro mordio al hombre" y "el hombre mordio al perro"
# darian los mismos vectores, solo en otro orden.

orden = np.array([3, 0, 4, 1, 2])
s_normal = mt.atencion(x, bloque_chico, H, causal=False).data
s_desordenada = mt.atencion(Tensor(x.data[:, orden]), bloque_chico, H, causal=False).data
dif = np.abs(s_desordenada - s_normal[:, orden]).max()
print(f"\nPARTE 4: salida(desordenar x) vs desordenar(salida x): diferencia maxima = {dif:.1e}")


# ============================================================
# PARTE 5: LAYER NORM
# ============================================================
# Cada vector se normaliza a media 0 y desviacion 1 (como la
# normalizacion de entradas de la 1.4, pero dentro de la red y por vector).

v = Tensor(rng.standard_normal((1, 1, D)) * 7 + 3)
vn = mt.layer_norm(v, Tensor(np.ones(D)), Tensor(np.zeros(D)))
print(f"\nPARTE 5: antes media={v.data.mean():.2f} desv={v.data.std():.2f}"
      f" | despues media={vn.data.mean():.2f} desv={vn.data.std():.2f}")


# ============================================================
# PARTE 6: GRADIENT CHECKING DEL BLOQUE COMPLETO
# ============================================================
# En la 4.1 se verifico cada operacion por separado. Ahora verificamos el
# bloque entero (atencion + mlp + normas + residuales), en 3 numeros de
# cada matriz de parametros.

objetivo = rng.standard_normal((B, T, D))


def perdida_bloque():
    return (mt.bloque(x, bloque_chico, H) * Tensor(objetivo)).sum()


parametros_en_cero(bloque_chico.values())
perdida_bloque().backward()
peor = 0.0
for nombre, p in bloque_chico.items():
    for _ in range(3):
        i = tuple(rng.integers(0, n) for n in p.shape)
        viejo = p.data[i]
        p.data[i] = viejo + 1e-5
        mas = perdida_bloque().data.item()
        p.data[i] = viejo - 1e-5
        menos = perdida_bloque().data.item()
        p.data[i] = viejo
        numerico = (mas - menos) / 2e-5
        peor = max(peor, abs(numerico - p.grad[i]) / max(abs(numerico), abs(p.grad[i]), 1e-8))
print(f"\nPARTE 6: gradient checking del bloque, peor error relativo = {peor:.1e}")


# ============================================================
# PARTE 7: ENTRENAR UNA TAREA QUE EXIGE POSICION
# ============================================================
# Secuencias de 16 digitos al azar. En cada posicion t >= 3, el modelo
# debe decir que digito habia en t - 3:
#     entrada:  4 7 1 9 0 3 ...
#     objetivo: . . . 4 7 1 ...
# No se puede resolver mirando el contenido: hay que saber DONDE esta
# cada digito. Comparamos el modelo con y sin embeddings de posicion.

V, T, RETARDO = 10, 16, 3
D, H, N_BLOQUES, PASOS = 32, 4, 1, 600


def lote(n):
    x = rng.integers(0, V, (n, T))
    return x, x[:, :T - RETARDO]


def entrenar(posiciones):
    modelo = mt.init_modelo(V, D, H, N_BLOQUES, T, posiciones=posiciones)
    params = mt.parametros(modelo)
    optimizador = Adam(params, tasa=3e-3)
    historial = []
    inicio = time.time()
    for paso in range(PASOS):
        x, y = lote(32)
        parametros_en_cero(params)
        logits = mt.forward(modelo, x)                 # (32, 16, 10)
        L = cross_entropy(logits[:, RETARDO:], y)      # solo las posiciones con respuesta
        L.backward()
        optimizador.paso()
        historial.append(L.data.item())
    x, y = lote(1000)
    exactitud = np.mean(mt.forward(modelo, x).data[:, RETARDO:].argmax(-1) == y)
    print(f"  posiciones={posiciones!s:<5} | parametros={mt.contar(modelo)} | "
          f"perdida final={np.mean(historial[-50:]):.3f} | exactitud={exactitud:.1%} | "
          f"{time.time() - inicio:.0f} s")
    return modelo, historial


print("\nPARTE 7: copiar el digito de 3 posiciones atras")
modelo, hist_con = entrenar(posiciones=True)
_, hist_sin = entrenar(posiciones=False)


# ============================================================
# PARTE 8: GRAFICOS
# ============================================================
ejemplo = rng.integers(0, V, (1, T))
alfas = []
mt.forward(modelo, ejemplo, guardar=alfas)

fig = plt.figure(figsize=(14, 6.5))
ax = fig.add_subplot(2, 3, (1, 4))
media = lambda h: np.convolve(h, np.ones(20) / 20, mode="valid")
ax.plot(media(hist_con), color="#EC4899", label="con posiciones")
ax.plot(media(hist_sin), color="#6B7280", label="sin posiciones")
ax.axhline(np.log(V), color="#9CA3AF", linestyle="--", linewidth=1)
ax.set_title("Perdida (promedio movil de 20 pasos)")
ax.set_xlabel("Paso")
ax.legend()
ax.grid(alpha=0.3)

for cabeza in range(H):
    ax = fig.add_subplot(2, 3, [2, 3, 5, 6][cabeza])
    ax.imshow(alfas[0][0, cabeza], cmap="Purples", vmin=0, vmax=1)
    ax.set_title(f"Cabeza {cabeza + 1}", fontsize=10)
    ax.set_xticks(range(T))
    ax.set_xticklabels(ejemplo[0], fontsize=7)
    ax.set_yticks(range(T))
    ax.set_yticklabels(ejemplo[0], fontsize=7)
fig.text(0.68, 0.01, "Pesos de atencion (fila = posicion que mira, columna = posicion mirada)",
         ha="center", fontsize=9)

plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("20_transformer_bloque.png", dpi=120)
print("\nGrafico guardado en: 20_transformer_bloque.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Auto-atencion: cada posicion es consulta, clave y valor; la mascara
   causal le impide mirar hacia adelante
2. Varias cabezas = varias atenciones en paralelo con vectores mas chicos;
   se juntan y se mezclan con Wo
3. La atencion no conoce el orden: hace falta sumar un embedding de posicion
4. Layer norm + conexiones residuales (x = x + f(x)) mantienen estables
   las activaciones y los gradientes al apilar bloques
5. Bloque = atencion (mezcla informacion ENTRE posiciones) + mlp (procesa
   CADA posicion por separado)

EXPERIMENTOS SUGERIDOS:
- Pon N_BLOQUES = 4 y llama a mt.forward(modelo, x, residual=False,
  norma=False) en entrenar(): sigue aprendiendo?
- Cambia RETARDO a 8, o H a 1: que pasa con la exactitud?
- Quita la mascara (causal=False en mt.bloque) y repite la Parte 3
""")
