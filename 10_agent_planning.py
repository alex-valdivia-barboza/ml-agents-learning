"""
LECCION 2.4 - PLANIFICACION (ESTILO STRIPS)
===========================================

Objetivo: que el agente razone sobre ACCIONES y sus EFECTOS en un mundo
descrito con hechos, no con coordenadas.

En las lecciones 2.2 y 2.3 el estado era una posicion (fila, col) y las
acciones eran "moverse". Aqui el estado es un CONJUNTO DE HECHOS, como
("sobre", "A", "B") = "el bloque A esta sobre el B", y cada accion dice:
  - PRECONDICIONES: que hechos tienen que ser ciertos para poder hacerla
  - EFECTOS: que hechos agrega y cuales elimina

Es el formato STRIPS (Stanford, 1971), base de los planificadores.

Problema: el "mundo de los bloques". Hay bloques sobre una mesa; se
puede mover un bloque que no tenga nada encima a la mesa o sobre otro
bloque libre. Queremos pasar de una torre a otra.

La busqueda es la misma de 2.2 (BFS) y 2.3 (A*): solo cambia que es un
estado y como se generan los sucesores.
"""

from collections import deque
from itertools import count, permutations
import heapq
import time

import matplotlib.pyplot as plt


# ============================================================
# PARTE 1: LAS ACCIONES (precondiciones y efectos)
# ============================================================
# Un estado es un frozenset de hechos (tuplas). frozenset es un set que
# no se puede modificar: asi puede usarse como clave de diccionario y
# guardarse en el conjunto de visitados.

class Accion:
    def __init__(self, nombre, precondiciones, agregar, eliminar):
        self.nombre = nombre
        self.precondiciones = frozenset(precondiciones)
        self.agregar = frozenset(agregar)
        self.eliminar = frozenset(eliminar)

    def aplicable(self, estado):
        return self.precondiciones <= estado                 # todas las precondiciones estan en el estado

    def aplicar(self, estado):
        return (estado - self.eliminar) | self.agregar       # quitar y agregar hechos

    def __repr__(self):
        return self.nombre


def crear_acciones(bloques):
    """Instancia las acciones para todos los bloques (grounding)."""
    acciones = []
    for x in bloques:
        for y in bloques:
            if x == y:
                continue
            # mover x desde la mesa hasta encima de y
            acciones.append(Accion(f"apilar({x} sobre {y})",
                                   [("en_mesa", x), ("libre", x), ("libre", y)],
                                   [("sobre", x, y)],
                                   [("en_mesa", x), ("libre", y)]))
            # mover x desde encima de y hasta la mesa
            acciones.append(Accion(f"bajar({x} de {y})",
                                   [("sobre", x, y), ("libre", x)],
                                   [("en_mesa", x), ("libre", y)],
                                   [("sobre", x, y)]))
            for z in bloques:
                if z in (x, y):
                    continue
                # mover x desde encima de y hasta encima de z
                acciones.append(Accion(f"mover({x} de {y} a {z})",
                                       [("sobre", x, y), ("libre", x), ("libre", z)],
                                       [("sobre", x, z), ("libre", y)],
                                       [("sobre", x, y), ("libre", z)]))
    return acciones


# ============================================================
# PARTE 2: EL PROBLEMA (la anomalia de Sussman)
# ============================================================
#   Inicio:  C          Meta:  A
#            A    B            B
#           ------             C
#                             ------

BLOQUES = ["A", "B", "C"]
INICIAL = frozenset([("sobre", "C", "A"), ("en_mesa", "A"), ("en_mesa", "B"),
                     ("libre", "C"), ("libre", "B")])
META = frozenset([("sobre", "A", "B"), ("sobre", "B", "C")])

ACCIONES = crear_acciones(BLOQUES)
print(f"Bloques: {BLOQUES} -> {len(ACCIONES)} acciones instanciadas")
print("Aplicables en el estado inicial:", [a for a in ACCIONES if a.aplicable(INICIAL)])


# ============================================================
# PARTE 3: PLANIFICACION HACIA ADELANTE (forward chaining) CON BFS
# ============================================================
# Desde el estado inicial, aplicar acciones hasta llegar a un estado que
# contenga todos los hechos de la meta. Es BFS de la 2.2 casi sin cambios.

def planificar_bfs(inicial, meta, acciones):
    frontera = deque([inicial])
    padre = {inicial: (None, None)}            # estado -> (estado anterior, accion)
    while frontera:
        estado = frontera.popleft()
        if meta <= estado:                     # la meta es un subconjunto del estado
            return reconstruir(padre, estado), len(padre)
        for accion in acciones:
            if accion.aplicable(estado):
                siguiente = accion.aplicar(estado)
                if siguiente not in padre:
                    padre[siguiente] = (estado, accion)
                    frontera.append(siguiente)
    return None, len(padre)


def reconstruir(padre, estado):
    plan = []
    while padre[estado][0] is not None:
        estado, accion = padre[estado]
        plan.append(accion)
    return plan[::-1]


plan, explorados = planificar_bfs(INICIAL, META, ACCIONES)
print(f"\n--- Hacia adelante (BFS) ---")
print(f"Plan de {len(plan)} acciones: {plan}  | estados descubiertos: {explorados}")


# ============================================================
# PARTE 4: PLANIFICACION HACIA ATRAS (backward chaining / regresion)
# ============================================================
# Desde la meta: buscar una accion que logre alguno de sus hechos sin
# destruir otro, y reemplazar la meta por "lo que haria falta ANTES"
# (regresion). Se termina cuando la sub-meta ya es cierta en el inicio.
#   regresar(meta, accion) = (meta - agregar) | precondiciones

def regresar(meta, accion):
    if not (accion.agregar & meta):            # la accion no aporta nada a la meta
        return None
    if accion.eliminar & meta:                 # la accion destruye parte de la meta
        return None
    return (meta - accion.agregar) | accion.precondiciones


def planificar_hacia_atras(inicial, meta, acciones):
    frontera = deque([meta])
    padre = {meta: (None, None)}
    while frontera:
        submeta = frontera.popleft()
        if submeta <= inicial:                 # ya se cumple en el estado inicial
            plan = []
            while padre[submeta][0] is not None:
                submeta, accion = padre[submeta]
                plan.append(accion)
            return plan, len(padre)            # se lee en orden: la primera accion va primero
        for accion in acciones:
            anterior = regresar(submeta, accion)
            if anterior is not None and anterior not in padre:
                padre[anterior] = (submeta, accion)
                frontera.append(anterior)
    return None, len(padre)


plan_atras, explorados_atras = planificar_hacia_atras(INICIAL, META, ACCIONES)
print(f"\n--- Hacia atras (regresion) ---")
print(f"Plan de {len(plan_atras)} acciones: {plan_atras}  | sub-metas descubiertas: {explorados_atras}")


# ============================================================
# PARTE 5: A* CON UNA HEURISTICA DE PLANIFICACION
# ============================================================
# h = cuantos hechos de la meta faltan. Cada accion mueve UN bloque, asi
# que agrega como mucho un hecho "sobre" de la meta: h no sobreestima.

def planificar_astar(inicial, meta, acciones):
    h = lambda estado: len(meta - estado)
    contador = count()
    frontera = [(h(inicial), next(contador), inicial)]
    g = {inicial: 0}
    padre = {inicial: (None, None)}
    cerrados = set()
    while frontera:
        _, _, estado = heapq.heappop(frontera)
        if estado in cerrados:
            continue
        cerrados.add(estado)
        if meta <= estado:
            return reconstruir(padre, estado), len(padre)
        for accion in acciones:
            if accion.aplicable(estado):
                siguiente = accion.aplicar(estado)
                if siguiente not in g or g[estado] + 1 < g[siguiente]:
                    g[siguiente] = g[estado] + 1
                    padre[siguiente] = (estado, accion)
                    heapq.heappush(frontera, (g[siguiente] + h(siguiente), next(contador), siguiente))
    return None, len(padre)


# ============================================================
# PARTE 6: ESCALAR: MAS BLOQUES
# ============================================================
# Inicio: todos los bloques apilados en orden (A sobre B sobre C ...).
# Meta:   la torre invertida.

def torre(orden):
    hechos = [("en_mesa", orden[-1]), ("libre", orden[0])]
    hechos += [("sobre", a, b) for a, b in zip(orden, orden[1:])]
    return frozenset(hechos)


print("\n--- Escalando: invertir una torre de n bloques ---")
tabla = []
for n in [3, 4, 5, 6]:
    bloques = [chr(ord("A") + i) for i in range(n)]
    acciones = crear_acciones(bloques)
    inicial = torre(bloques)
    meta = frozenset(h for h in torre(bloques[::-1]) if h[0] == "sobre")
    fila = {"n": n}
    for nombre, planificador in [("BFS", planificar_bfs), ("A*", planificar_astar)]:
        t0 = time.perf_counter()
        p, e = planificador(inicial, meta, acciones)
        fila[nombre] = (len(p), e, time.perf_counter() - t0)
    tabla.append(fila)
    print(f"n={n} | acciones={len(acciones):3d} | BFS: plan {fila['BFS'][0]}, {fila['BFS'][1]:6d} estados "
          f"| A*: plan {fila['A*'][0]}, {fila['A*'][1]:6d} estados")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================
# (1) el plan de la anomalia de Sussman, paso a paso
# (2) estados explorados segun la cantidad de bloques

def dibujar_estado(ax, estado, x0, titulo):
    sobre = {h[1]: h[2] for h in estado if h[0] == "sobre"}
    base = [h[1] for h in estado if h[0] == "en_mesa"]
    for i, b in enumerate(sorted(base)):
        pila, actual = [b], b
        while actual in sobre.values():
            actual = [k for k, v in sobre.items() if v == actual][0]
            pila.append(actual)
        for nivel, bloque in enumerate(pila):
            ax.add_patch(plt.Rectangle((x0 + i * 1.1, nivel), 1, 1, color="#3B82F6", ec="white"))
            ax.text(x0 + i * 1.1 + 0.5, nivel + 0.5, bloque, ha="center", va="center", color="white", fontweight="bold")
    ax.text(x0 + 1.6, -0.6, titulo, ha="center", fontsize=8)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
estado = INICIAL
dibujar_estado(ax1, estado, 0, "inicio")
for i, accion in enumerate(plan):
    estado = accion.aplicar(estado)
    dibujar_estado(ax1, estado, (i + 1) * 4.2, str(accion))
ax1.plot([-0.3, (len(plan) + 1) * 4.2 - 0.6], [0, 0], color="black")
ax1.set_xlim(-0.5, (len(plan) + 1) * 4.2)
ax1.set_ylim(-1, 3.5)
ax1.axis("off")
ax1.set_title("Plan para la anomalia de Sussman")

ns = [f["n"] for f in tabla]
ax2.plot(ns, [f["BFS"][1] for f in tabla], marker="o", color="#F59E0B", label="BFS")
ax2.plot(ns, [f["A*"][1] for f in tabla], marker="o", color="#10B981", label="A*")
ax2.set_yscale("log")
ax2.set_xlabel("Bloques")
ax2.set_ylabel("Estados descubiertos (log)")
ax2.set_title("Invertir una torre de n bloques")
ax2.legend()

plt.tight_layout()
plt.savefig("10_agent_planning.png", dpi=120)
print("\nGrafico guardado en: 10_agent_planning.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un estado puede describirse como un conjunto de hechos
2. Una accion STRIPS tiene precondiciones (cuando se puede hacer) y efectos
   (que hechos agrega y cuales elimina)
3. Planificar hacia adelante = buscar (BFS, A*) en el espacio de estados
   generado por las acciones aplicables
4. Planificar hacia atras = partir de la meta y preguntar "que deberia
   ser cierto antes?" (regresion)
5. La cantidad de estados crece explosivamente con los objetos: hacen
   falta heuristicas

EXPERIMENTOS SUGERIDOS:
- Cambia META por la torre C sobre B sobre A: cuantas acciones hacen falta?
- Resuelve la meta de a una sub-meta por vez (primero A sobre B, despues
  B sobre C, sin deshacer la anterior): por que falla?
- Agrega una accion "cargar" con un hecho ("mano_vacia") y ve como cambia
  el numero de estados
""")
