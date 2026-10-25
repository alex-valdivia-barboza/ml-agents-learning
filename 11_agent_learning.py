"""
LECCION 2.5 - Q-LEARNING BASICO
===============================

Objetivo: un agente que APRENDE a decidir a partir de la experiencia.

En las lecciones 2.2 a 2.4 el agente conocia el mapa (o las acciones y
sus efectos) y buscaba un plan antes de actuar. Aqui NO conoce nada:
solo puede actuar y observar que pasa. Despues de cada accion el entorno
le devuelve:
  - el nuevo estado
  - una RECOMPENSA (un numero: positivo = bueno, negativo = malo)

El problema: "el acantilado" (cliff walking). Una cuadricula de 4x10.
El agente empieza abajo a la izquierda (S) y tiene que llegar abajo a la
derecha (G). Entre ambos, la fila de abajo es un acantilado (C):
  - cada paso cuesta -1 (apurarse es bueno)
  - caer al acantilado cuesta -100 y lo devuelve al inicio

Q-learning aprende una tabla Q(estado, accion) = "cuanta recompensa
total espero si hago esta accion aqui y despues actuo lo mejor posible".
"""

import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(42)
np.random.seed(42)


# ============================================================
# PARTE 1: EL ENTORNO (una caja negra para el agente)
# ============================================================

FILAS, COLUMNAS = 4, 10
INICIO, META = (3, 0), (3, 9)
ACANTILADO = {(3, c) for c in range(1, 9)}
ACCIONES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # arriba, abajo, izquierda, derecha
NOMBRES = ["arriba", "abajo", "izquierda", "derecha"]


class Acantilado:
    def reiniciar(self):
        self.pos = INICIO
        return self.pos

    def paso(self, accion):
        """Aplica una accion. Devuelve (nuevo_estado, recompensa, terminado)."""
        df, dc = ACCIONES[accion]
        f = min(max(self.pos[0] + df, 0), FILAS - 1)   # chocar con el borde = quedarse
        c = min(max(self.pos[1] + dc, 0), COLUMNAS - 1)
        self.pos = (f, c)
        if self.pos in ACANTILADO:
            self.pos = INICIO                           # se cae: vuelve al inicio
            return self.pos, -100, False
        if self.pos == META:
            return self.pos, -1, True
        return self.pos, -1, False


# ============================================================
# PARTE 2: LA TABLA Q Y LA ELECCION DE ACCION (epsilon-greedy)
# ============================================================
# Q es un array de (4, 10, 4): un valor por cada estado (fila, col) y
# cada una de las 4 acciones. Empieza en 0: el agente no sabe nada.
#
# epsilon-greedy: con probabilidad epsilon EXPLORA (accion al azar);
# si no, EXPLOTA lo que sabe (la accion con mayor Q).

def elegir_accion(Q, estado, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(ACCIONES))          # explorar
    valores = Q[estado]
    mejores = np.flatnonzero(valores == valores.max())  # puede haber empates
    return int(random.choice(mejores))                  # explotar


# ============================================================
# PARTE 3: Q-LEARNING
# ============================================================
# Despues de cada paso (s, a) -> (s', r):
#   objetivo = r + gamma * max_a' Q(s', a')      (lo que "deberia" valer)
#   Q(s, a) += alfa * (objetivo - Q(s, a))       (acercarse un poco)
# Si s' es terminal, no hay futuro: objetivo = r.

def entrenar(episodios=500, alfa=0.5, gamma=1.0, epsilon=0.1, max_pasos=500):
    Q = np.zeros((FILAS, COLUMNAS, len(ACCIONES)))
    entorno = Acantilado()
    recompensas, caidas = [], []
    for episodio in range(episodios):
        estado = entorno.reiniciar()
        total, n_caidas = 0, 0
        for _ in range(max_pasos):
            accion = elegir_accion(Q, estado, epsilon)
            siguiente, recompensa, terminado = entorno.paso(accion)
            objetivo = recompensa if terminado else recompensa + gamma * Q[siguiente].max()
            Q[estado][accion] += alfa * (objetivo - Q[estado][accion])
            total += recompensa
            n_caidas += recompensa == -100
            estado = siguiente
            if terminado:
                break
        recompensas.append(total)
        caidas.append(n_caidas)
    return Q, recompensas, caidas


print("--- Entrenando 500 episodios (alfa=0.5, gamma=1.0, epsilon=0.1) ---")
Q, recompensas, caidas = entrenar()
for inicio in [0, 10, 50, 100, 200, 499]:
    print(f"Episodio {inicio:3d}: recompensa total = {recompensas[inicio]:5d} | caidas = {caidas[inicio]}")


# ============================================================
# PARTE 4: LA POLITICA APRENDIDA
# ============================================================
# Politica = en cada estado, la accion con mayor Q (sin explorar).

def seguir_politica(Q, max_pasos=50):
    entorno = Acantilado()
    estado = entorno.reiniciar()
    camino, total = [estado], 0
    for _ in range(max_pasos):
        estado, recompensa, terminado = entorno.paso(int(np.argmax(Q[estado])))
        camino.append(estado)
        total += recompensa
        if terminado:
            break
    return camino, total


camino, total = seguir_politica(Q)
print(f"\nPolitica aprendida: {len(camino) - 1} pasos, recompensa {total}")
print(f"Camino: {camino}")
print("Q(inicio) =", "  ".join(f"{n}: {v:.1f}" for n, v in zip(NOMBRES, Q[INICIO])))


# ============================================================
# PARTE 5: EXPLORACION VS EXPLOTACION
# ============================================================

print("\n--- Efecto de epsilon (promedio de los ultimos 100 episodios) ---")
por_epsilon = {}
for eps in [0.0, 0.1, 0.3]:
    random.seed(0)
    Qe, re, ce = entrenar(epsilon=eps)
    ruta, tot = seguir_politica(Qe)
    por_epsilon[eps] = re
    print(f"epsilon={eps:<4} | recompensa media={np.mean(re[-100:]):7.1f} | "
          f"politica final: {len(ruta) - 1} pasos, recompensa {tot}")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================

def media_movil(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="valid")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})

for eps, color in zip(por_epsilon, ["#9CA3AF", "#10B981", "#F59E0B"]):
    ax1.plot(media_movil(por_epsilon[eps]), color=color, label=f"epsilon={eps}")
ax1.set_ylim(-300, 0)
ax1.set_title("Recompensa por episodio (media movil de 10)")
ax1.set_xlabel("Episodio")
ax1.set_ylabel("Recompensa total")
ax1.legend()

valores = Q.max(axis=2)
for (f, c) in ACANTILADO:
    valores[f, c] = np.nan
im = ax2.imshow(valores, cmap="viridis")
fig.colorbar(im, ax=ax2, label="max Q (recompensa esperada)")
flechas = ["^", "v", "<", ">"]
for f in range(FILAS):
    for c in range(COLUMNAS):
        if (f, c) in ACANTILADO:
            ax2.text(c, f, "C", ha="center", va="center", color="#EF4444", fontweight="bold")
        elif (f, c) == META:
            ax2.text(c, f, "G", ha="center", va="center", color="white", fontweight="bold")
        else:
            ax2.text(c, f, flechas[int(np.argmax(Q[f, c]))], ha="center", va="center", color="white")
ax2.plot([c for _, c in camino], [f for f, _ in camino], color="#EF4444", linewidth=2, alpha=0.6)
ax2.set_title("Politica aprendida (flechas) y valor de cada estado")
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig("11_agent_learning.png", dpi=120)
print("\nGrafico guardado en: 11_agent_learning.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Aprendizaje por refuerzo: el agente no conoce el entorno; aprende de
   las recompensas que recibe al actuar
2. Q(s, a) estima la recompensa total futura de hacer a en s
3. Regla de Q-learning: Q += alfa * (r + gamma * max Q(s') - Q)
   (como en las redes: acercar la estimacion a un objetivo, de a poco)
4. epsilon-greedy equilibra explorar (probar) y explotar (usar lo sabido)
5. La politica final es elegir en cada estado la accion de mayor Q

EXPERIMENTOS SUGERIDOS:
- Cambia alfa a 0.05 y a 1.0: cuantos episodios tarda en aprender?
- Usa gamma = 0.9: cambia el camino aprendido?
- Con epsilon = 0.0 el agente nunca elige al azar: por que igual
  encuentra el camino? (pista: Q empieza en 0 y toda recompensa es negativa)
""")
