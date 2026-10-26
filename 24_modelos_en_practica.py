"""
LECCION 4.6 - MODELOS GRANDES EN LA PRACTICA
============================================

Objetivo: entender, con el mini-GPT como laboratorio, las ideas que
explican como se construyen y se usan los modelos grandes como Claude.

  1. PREENTRENAMIENTO: predecir el siguiente token en muchisimo texto
     (lo que hizo la 4.3 con el Quijote)
  2. AJUSTE FINO: seguir entrenando el mismo modelo, poco, con datos de
     otro tipo. Lo aprendido antes no se tira: se aprovecha
  3. PEDIDOS Y RESPUESTAS: plantear una tarea como texto a continuar, y
     ajustar el modelo con ejemplos de ese formato (asi se convierte un
     "completador de texto" en un asistente)
  4. VENTANA DE CONTEXTO: cuanto texto puede leer de una vez, y por que
     agrandarla cuesta tanto
  5. ESCALA: cuanto computo hace falta, y por que importa elegir modelo

Usa el mini-GPT entrenado en la leccion 4.3 (data/mini_gpt.npz).
AVISO: tarda unos minutos.
"""

import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import mini_transformer as mt
from corpus_es import RESENAS
from mini_autograd import Adam, cross_entropy, parametros_en_cero

rng = np.random.default_rng(42)

if not os.path.exists("data/mini_gpt.npz"):
    raise SystemExit("Falta data/mini_gpt.npz: ejecuta primero 21_mini_gpt.py (leccion 4.3)")
preentrenado, archivo = mt.cargar("data/mini_gpt.npz")
caracteres = [str(c) for c in archivo["caracteres"]]
a_indice = {ch: i for i, ch in enumerate(caracteres)}
T = preentrenado["P"].shape[0]


def codificar(texto):
    return [a_indice[ch] for ch in texto]


def copia_del_preentrenado():
    return copy.deepcopy(preentrenado)                  # el original no se toca


def modelo_al_azar(semilla):
    return mt.init_modelo(len(caracteres), 64, 4, 2, T, semilla=semilla)


# Validacion cruzada: las 80 resenas en 4 grupos de 20. Cada vez se ajusta
# con 60 y se evalua con las 20 que el ajuste no vio.
orden = rng.permutation(len(RESENAS))
grupos = np.array_split(orden, 4)


# ============================================================
# PARTE 1: AJUSTE FINO DE LENGUAJE
# ============================================================
# El mini-GPT solo leyo el Quijote. Las resenas son otro mundo: "el envio
# llego rapido", "la bateria dura poco". Lo ajustamos para que prediga
# texto de resenas y medimos la perplejidad en resenas nuevas. Comparamos:
#   - sin ajuste:   el preentrenado tal cual
#   - preentrenado: el preentrenado + ajuste con 60 resenas
#   - desde cero:   pesos al azar + el mismo ajuste (solo conoce las resenas)

PASOS_AJUSTE, LOTE, TRAMO = 150, 16, 48


def unir(resenas):
    return codificar(". ".join(t.lower() for t, _ in resenas))


def ajustar_lenguaje(modelo, datos):
    params = mt.parametros(modelo)
    optimizador = Adam(params, tasa=1e-3)               # tasa mas chica que al preentrenar
    for paso in range(PASOS_AJUSTE):
        inicios = rng.integers(0, len(datos) - TRAMO - 1, LOTE)
        x = np.array([datos[i:i + TRAMO] for i in inicios])
        y = np.array([datos[i + 1:i + TRAMO + 1] for i in inicios])
        parametros_en_cero(params)
        cross_entropy(mt.forward(modelo, x), y).backward()
        optimizador.paso()


def perplejidad(modelo, datos):
    # tramos seguidos de T caracteres: cada caracter se predice una vez
    perdidas = []
    for i in range(0, len(datos) - 1, T):
        x = np.array([datos[i:i + T]])[:, :len(datos) - 1 - i]
        y = np.array([datos[i + 1:i + T + 1]])
        perdidas.append(cross_entropy(mt.forward(modelo, x), y).data.item() * y.size)
    return np.exp(sum(perdidas) / (len(datos) - 1))


print("PARTE 1: perplejidad en resenas que el ajuste no vio")
pp = {"sin ajuste": [], "preentrenado": [], "desde cero": []}
inicio = time.time()
for g, prueba in enumerate(grupos):
    entrenamiento = unir([RESENAS[i] for i in orden if i not in prueba])
    test = unir([RESENAS[i] for i in prueba])
    pp["sin ajuste"].append(perplejidad(preentrenado, test))
    for nombre, crear in [("preentrenado", copia_del_preentrenado),
                          ("desde cero", lambda: modelo_al_azar(g + 1))]:
        modelo = crear()
        ajustar_lenguaje(modelo, entrenamiento)
        pp[nombre].append(perplejidad(modelo, test))
        if nombre == "preentrenado" and g == 0:
            ajustado_lenguaje = modelo
    print(f"  grupo {g + 1}/4 listo ({time.time() - inicio:.0f} s)")
for nombre, valores in pp.items():
    print(f"  {nombre:<13} {np.mean(valores):6.2f}   (por grupo: {np.round(valores, 1)})")


def continuar(modelo, texto, n=60):
    for _ in range(n):
        logits = mt.forward(modelo, np.array([codificar(texto[-T:])])).data[0, -1]
        probs = np.exp((logits - logits.max()) / 0.7)
        texto += caracteres[rng.choice(len(caracteres), p=probs / probs.sum())]
    return texto


print("Continuar 'el envio llego ' (temperatura 0.7):")
print("  sin ajuste:  ", repr(continuar(preentrenado, "el envio llego ")))
print("  con ajuste:  ", repr(continuar(ajustado_lenguaje, "el envio llego ")))


# ============================================================
# PARTE 2: UNA TAREA COMO PEDIDO Y RESPUESTA
# ============================================================
# Un modelo de lenguaje solo sabe continuar texto. Para que clasifique
# opiniones, escribimos cada ejemplo como un texto que termina en la
# respuesta:
#     "la comida estaba deliciosa ¿buena? s"
#     "el envio llego roto ¿buena? n"
# Despues de "¿buena? " el modelo tiene que escribir "s" o "n". El ajuste
# solo calcula la perdida en ese caracter: no queremos que aprenda a
# escribir resenas, sino a responder.

def formatear(texto, etiqueta=None):
    pedido = texto.lower() + " ¿buena? "
    return pedido if etiqueta is None else pedido + ("s" if etiqueta == 1 else "n")


def responder(modelo, texto):
    """Compara el puntaje de 's' contra el de 'n' en la ultima posicion."""
    logits = mt.forward(modelo, np.array([codificar(formatear(texto))[-T:]])).data[0, -1]
    return 1 if logits[a_indice["s"]] > logits[a_indice["n"]] else 0


def ajustar_respuestas(modelo, ejemplos):
    params = mt.parametros(modelo)
    optimizador = Adam(params, tasa=1e-3)
    for paso in range(PASOS_AJUSTE):
        elegidos = rng.choice(len(ejemplos), LOTE)
        seqs = [codificar(formatear(*ejemplos[i]))[-T:] for i in elegidos]
        largo = max(len(s) for s in seqs)
        # completar a la izquierda con espacios para que todas midan lo mismo
        x = np.array([[a_indice[" "]] * (largo - len(s)) + s for s in seqs])
        parametros_en_cero(params)
        logits = mt.forward(modelo, x[:, :-1])
        cross_entropy(logits[:, -1], x[:, -1]).backward()   # solo la ultima posicion
        optimizador.paso()


print("\nEjemplo:", repr(formatear(*RESENAS[0])))
texto = formatear(RESENAS[0][0])
for _ in range(25):
    logits = mt.forward(preentrenado, np.array([codificar(texto[-T:])])).data[0, -1]
    texto += caracteres[int(np.argmax(logits))]
print("Sin ajuste, el preentrenado continua asi:", repr(texto))

aciertos = {"sin ajuste": [], "preentrenado": [], "desde cero": []}
respuestas_s = {"preentrenado": [], "desde cero": []}
inicio = time.time()
for g, prueba in enumerate(grupos):
    entrenamiento = [RESENAS[i] for i in orden if i not in prueba]
    test = [RESENAS[i] for i in prueba]
    aciertos["sin ajuste"] += [responder(preentrenado, t) == e for t, e in test]
    for nombre, crear in [("preentrenado", copia_del_preentrenado),
                          ("desde cero", lambda: modelo_al_azar(g + 1))]:
        modelo = crear()
        ajustar_respuestas(modelo, entrenamiento)
        r = [responder(modelo, t) for t, _ in test]
        aciertos[nombre] += [a == e for a, (_, e) in zip(r, test)]
        respuestas_s[nombre] += r
    print(f"  grupo {g + 1}/4 listo ({time.time() - inicio:.0f} s)")
print("PARTE 2: exactitud en resenas que el ajuste no vio (80 en total)")
for nombre, a in aciertos.items():
    extra = f" | respondio 's' en el {np.mean(respuestas_s[nombre]):.0%}" if nombre in respuestas_s else ""
    print(f"  {nombre:<13} {np.mean(a):.0%}{extra}")


# ============================================================
# PARTE 3: CUANTO CUESTA LA VENTANA DE CONTEXTO
# ============================================================
# La atencion compara todas las posiciones con todas: su matriz de pesos
# tiene T x T numeros por cabeza. Medimos el tiempo de un forward con
# ventanas cada vez mas largas (modelo al azar, mismo tamano).

largos = [32, 64, 128, 256, 512]
grande = mt.init_modelo(len(caracteres), 64, 4, 2, max(largos))
tiempos = []
for L_ in largos:
    x = rng.integers(0, len(caracteres), (8, L_))
    t0 = time.perf_counter()
    for _ in range(3):
        mt.forward(grande, x)
    tiempos.append((time.perf_counter() - t0) / 3)
print("\nPARTE 3: tiempo de un forward (8 secuencias) segun el largo de la ventana")
for L_, t in zip(largos, tiempos):
    print(f"  T = {L_:4d}: {t * 1000:7.1f} ms | pesos de atencion por secuencia: {4 * 2 * L_ * L_:,} numeros")


# ============================================================
# PARTE 4: LA ESCALA
# ============================================================
# Regla aproximada muy usada: entrenar cuesta unas 6 operaciones por
# parametro y por token (2 en el forward, 4 en el backward).

n_params = mt.contar(preentrenado)
tokens_vistos = 1500 * 32 * 64                      # pasos x lote x ventana de la 4.3
ops_mini = 6 * n_params * tokens_vistos
ops_gpt3 = 6 * 175e9 * 300e9                         # GPT-3 (2020): datos publicados
print(f"\nPARTE 4: mini-GPT: {n_params:,} parametros, {tokens_vistos:,} tokens -> {ops_mini:.1e} operaciones")
print(f"         GPT-3:    175,000,000,000 parametros, 300,000,000,000 tokens -> {ops_gpt3:.1e} operaciones")
print(f"         {ops_gpt3 / ops_mini:.0e} veces mas computo")


# ============================================================
# PARTE 5: GRAFICOS
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
colores = ["#D1D5DB", "#EC4899", "#6B7280"]
nombres = list(pp)
ax1.bar(nombres, [np.mean(pp[n]) for n in nombres], color=colores)
ax1.set_ylabel("Perplejidad en resenas nuevas")
ax1.set_title("Ajuste fino de lenguaje (menos es mejor)")

ax2.bar(nombres, [np.mean(aciertos[n]) * 100 for n in nombres], color=colores)
ax2.axhline(50, color="#9CA3AF", linestyle="--", linewidth=1)
ax2.set_ylabel("Exactitud en test (%)")
ax2.set_title("Pedido y respuesta: clasificar resenas")

ax3.plot(largos, np.array(tiempos) * 1000, "o-", color="#EC4899")
ax3.set_title("Costo de la ventana de contexto")
ax3.set_xlabel("Largo de la ventana T")
ax3.set_ylabel("ms por forward")
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("24_modelos_en_practica.png", dpi=120)
print("\nGrafico guardado en: 24_modelos_en_practica.png")
print("Abriendo ventana con el grafico... (cierrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Preentrenar (predecir el siguiente token en mucho texto) da una base
   que despues se adapta con pocos datos: ajuste fino
2. Un modelo de lenguaje resuelve tareas si se las plantea como texto a
   continuar; ajustarlo con pares pedido-respuesta lo vuelve un asistente,
   pero eso solo funciona si el preentrenamiento ya le dio lo necesario
3. La ventana de contexto cuesta caro: la atencion crece con T al cuadrado
4. El computo de entrenamiento crece con parametros x tokens; por eso
   conviene elegir el modelo mas chico que resuelva bien la tarea

EXPERIMENTOS SUGERIDOS:
- Cambia PASOS_AJUSTE a 30 y a 400: con cual se sobreajusta "desde cero"?
- En ajustar_respuestas, calcula la perdida en TODOS los caracteres
- Cambia el formato del pedido: "opinion: ... - s/n". Importa la redaccion?
""")
