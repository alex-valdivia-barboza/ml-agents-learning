"""
LECCION 4.3 - MINI-GPT
======================

Objetivo: entrenar un transformer pequeno que escribe "a lo Quijote",
caracter por caracter, y compararlo con la RNN de la leccion 3.5.

Es la misma tarea de la 3.5 (predecir el siguiente caracter), con la
arquitectura de la 4.2 en lugar de la RNN:
  - la RNN leia de a un caracter y resumia todo en su memoria h
  - el transformer ve una VENTANA de 64 caracteres y cada posicion
    puede mirar directamente cualquier caracter anterior de la ventana
  - gracias a la mascara causal, una sola pasada entrena 64 predicciones
    a la vez (una por posicion)

GPT significa "Generative Pre-trained Transformer": transformer (4.2),
pre-entrenado prediciendo el siguiente token (esta leccion), generativo
porque escribe repitiendo esa prediccion. Es lo mismo que hacen los
modelos grandes, con tokens en vez de caracteres (4.4) y millones de
veces mas parametros y texto (4.6).

AVISO: el entrenamiento tarda unos minutos (todo es NumPy en la CPU).
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import mini_transformer as mt
from corpus_es import cargar_quijote
from mini_autograd import Adam, cross_entropy, parametros_en_cero

rng = np.random.default_rng(42)


# ============================================================
# PARTE 1: LOS DATOS (los mismos de la 3.5)
# ============================================================

texto = cargar_quijote().lower()
texto = " ".join(texto.split())
caracteres = sorted(set(texto))
V = len(caracteres)
a_indice = {ch: i for i, ch in enumerate(caracteres)}
datos = np.array([a_indice[ch] for ch in texto])
corte = int(len(datos) * 0.95)
train, test = datos[:corte], datos[corte:]
print(f"Caracteres: {len(datos):,} | vocabulario: {V} simbolos")


# ============================================================
# PARTE 2: EL MODELO
# ============================================================
# Embeddings de 64 numeros, 4 cabezas, 2 bloques, ventana de 64.

D, H, N_BLOQUES, T = 64, 4, 2, 64
modelo = mt.init_modelo(V, D, H, N_BLOQUES, T)
params = mt.parametros(modelo)
print(f"Parametros: {mt.contar(modelo):,} (la RNN de la 3.5 tenia 32,189)")


# ============================================================
# PARTE 3: LOTES: CADA VENTANA TRAE 64 EJEMPLOS
# ============================================================
# Una ventana de 65 caracteres da la entrada (los primeros 64) y el
# objetivo (los mismos, corridos un lugar). La posicion t ve los
# caracteres 0..t y debe predecir el t+1.

def lote(fuente, n):
    inicios = rng.integers(0, len(fuente) - T - 1, n)
    x = np.stack([fuente[i:i + T] for i in inicios])
    y = np.stack([fuente[i + 1:i + T + 1] for i in inicios])
    return x, y


x, y = lote(train, 1)
print("Entrada :", repr("".join(caracteres[i] for i in x[0][:30])))
print("Objetivo:", repr("".join(caracteres[i] for i in y[0][:30])))

# Conjunto fijo de ventanas de test para medir siempre con lo mismo
x_test, y_test = lote(test, 200)


def perdida_test():
    perdidas = [cross_entropy(mt.forward(modelo, x_test[i:i + 50]), y_test[i:i + 50]).data.item()
                for i in range(0, len(x_test), 50)]
    return np.mean(perdidas)


# ============================================================
# PARTE 4: GENERAR TEXTO
# ============================================================
# Como en la 3.3 y la 3.5: predecir, sortear, agregar, repetir. La
# ventana solo admite T caracteres: si el texto es mas largo, se usan
# los ultimos T.

def generar(inicio, n, temperatura=0.8):
    tokens = [a_indice[ch] for ch in inicio]
    for _ in range(n):
        contexto = np.array([tokens[-T:]])
        logits = mt.forward(modelo, contexto).data[0, -1]       # solo importa la ultima posicion
        probs = np.exp((logits - logits.max()) / temperatura)
        probs /= probs.sum()
        tokens.append(rng.choice(V, p=probs))
    return "".join(caracteres[i] for i in tokens)


# ============================================================
# PARTE 5: ENTRENAR
# ============================================================

PASOS, LOTE = 1500, 32
optimizador = Adam(params, tasa=3e-3)
hist_train, hist_test = [], []
inicio = time.time()
for paso in range(PASOS + 1):
    if paso % 250 == 0:
        hist_test.append((paso, perdida_test()))
        print(f"\nPaso {paso:5d} | perdida test={hist_test[-1][1]:.3f} "
              f"(perplejidad {np.exp(hist_test[-1][1]):.2f}) | {time.time() - inicio:.0f} s")
        print("  ", generar("don quijote ", 80))
    if paso == PASOS:
        break
    optimizador.tasa = 3e-3 * min(1.0, 1.5 * (1 - paso / PASOS))   # bajar la tasa al final
    x, y = lote(train, LOTE)
    parametros_en_cero(params)
    L = cross_entropy(mt.forward(modelo, x), y)
    L.backward()
    optimizador.paso()
    hist_train.append(L.data.item())


# El modelo entrenado se guarda en data/ para usarlo en la leccion 4.5
mt.guardar(modelo, "data/mini_gpt.npz", caracteres=np.array(caracteres))
print("\nModelo guardado en data/mini_gpt.npz")


# ============================================================
# PARTE 6: COMPARAR CON LA 3.5
# ============================================================
# La 3.5 midio en 20 000 caracteres seguidos del test; aqui usamos 200
# ventanas de 64. Para comparar, calculamos tambien los bigramas sobre
# estas mismas ventanas.

cuentas = np.ones((V, V))
np.add.at(cuentas, (train[:-1], train[1:]), 1)
probs_bigrama = cuentas / cuentas.sum(axis=1, keepdims=True)
ce_bigrama = -np.mean(np.log(probs_bigrama[x_test, y_test]))
ce_gpt = perdida_test()
print("\nPARTE 6: perplejidad en test")
print(f"  bigramas de caracteres: {np.exp(ce_bigrama):.2f}")
print(f"  mini-GPT:               {np.exp(ce_gpt):.2f}")
print("  (RNN de la 3.5:          6.66, medida en otro tramo del test)")

for temperatura in [0.5, 1.0]:
    print(f"\nTemperatura {temperatura}:")
    print("  ", generar("sancho ", 200, temperatura))


# ============================================================
# PARTE 7: GRAFICOS
# ============================================================
frase = "dijo sancho panza a su amo, don quijote de la mancha"
alfas = []
mt.forward(modelo, np.array([[a_indice[ch] for ch in frase]]), guardar=alfas)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
media = np.convolve(hist_train, np.ones(50) / 50, mode="valid")
ax1.plot(media, color="#EC4899", label="entrenamiento (promedio movil)")
ax1.plot([p for p, _ in hist_test], [l for _, l in hist_test], "o-", color="#3B82F6", label="test")
ax1.axhline(np.log(6.66), color="#10B981", linestyle="--", linewidth=1, label="RNN de la 3.5")
ax1.axhline(ce_bigrama, color="#6B7280", linestyle=":", linewidth=1, label="bigramas")
ax1.set_title("Perdida por caracter")
ax1.set_xlabel("Paso")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.imshow(alfas[-1][0].mean(axis=0), cmap="Purples")
ax2.set_xticks(range(len(frase)))
ax2.set_xticklabels(list(frase), fontsize=7)
ax2.set_yticks(range(len(frase)))
ax2.set_yticklabels(list(frase), fontsize=7)
ax2.set_title("Atencion del ultimo bloque (promedio de las cabezas)")

plt.tight_layout()
plt.savefig("21_mini_gpt.png", dpi=120)
print("\nGrafico guardado en: 21_mini_gpt.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un GPT es: embeddings + posiciones -> bloques transformer -> logits
   del siguiente token
2. Cada ventana de T caracteres entrena T predicciones a la vez, gracias
   a la mascara causal
3. Generar = predecir el siguiente, sortear, agregarlo y repetir, con una
   ventana de contexto de largo fijo
4. Con un numero parecido de pasos, el transformer supera a la RNN porque
   cada posicion accede directo a todo el contexto

EXPERIMENTOS SUGERIDOS:
- Cambia N_BLOQUES a 1 o a 4 (y mira el tiempo y la perplejidad)
- Cambia T a 16: cuanto empeora con menos contexto?
- Genera con un inicio que no esta en el libro ("la computadora ")
""")
