"""
LECCION 2.2 - BUSQUEDA EN ANCHURA (BFS)
=======================================

Objetivo: que el agente "piense antes de actuar". En la leccion 2.1 el
agente decidia paso a paso. Aqui conoce el mapa completo y, ANTES de
moverse, busca la secuencia de acciones que lo lleva a la meta.

Para eso el problema se formula como un ESPACIO DE ESTADOS:
  - estado:        la posicion (fila, col) del agente
  - acciones:      moverse arriba / abajo / izquierda / derecha
  - estado inicial: la S del mapa
  - estado meta:    la G del mapa

BFS (Breadth-First Search, busqueda en anchura) explora los estados por
"capas": primero todos los que estan a 1 paso, despues los que estan a
2, etc. Por eso, la primera vez que llega a la meta, lo hizo por el
camino mas corto.
"""

from collections import deque
import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(42)


# ============================================================
# PARTE 1: EL MAPA
# ============================================================
# '#' = pared, '.' = libre, 'S' = inicio, 'G' = meta (goal)

MAPA = [
    "############",
    "#S.....#...#",
    "#.####.#.#.#",
    "#.#....#.#.#",
    "#.#.####.#.#",
    "#...#....#.#",
    "###.#.##.#.#",
    "#.....#....#",
    "#.###.#.##.#",
    "#...#...#.G#",
    "############",
]


def leer_mapa(mapa):
    paredes = set()
    for f, fila in enumerate(mapa):
        for c, celda in enumerate(fila):
            if celda == "#":
                paredes.add((f, c))
            elif celda == "S":
                inicio = (f, c)
            elif celda == "G":
                meta = (f, c)
    return paredes, inicio, meta


PAREDES, INICIO, META = leer_mapa(MAPA)
print(f"Mapa de {len(MAPA)}x{len(MAPA[0])}, {len(PAREDES)} paredes")
print(f"Inicio: {INICIO}   Meta: {META}")


# ============================================================
# PARTE 2: EL ESPACIO DE ESTADOS
# ============================================================
# La funcion sucesores responde: "desde este estado, a que estados
# puedo llegar con una accion?". Es celdas_vecinas de la 2.1, pero
# descartando paredes en vez de bordes.

MOVIMIENTOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]       # arriba, abajo, izquierda, derecha


def sucesores(estado, paredes):
    f, c = estado
    return [(f + df, c + dc) for df, dc in MOVIMIENTOS if (f + df, c + dc) not in paredes]


print(f"Sucesores del inicio {INICIO}: {sucesores(INICIO, PAREDES)}")


# ============================================================
# PARTE 3: BFS
# ============================================================
# - frontera: cola FIFO (el primero que entra es el primero que sale)
#   con los estados descubiertos pendientes de explorar
# - padre:    para cada estado descubierto, desde cual se llego. Sirve
#   como conjunto de "ya visitados" y para reconstruir el camino
# - orden:    en que orden se expandio cada estado (para graficar)

def bfs(inicio, meta, paredes):
    frontera = deque([inicio])
    padre = {inicio: None}
    distancia = {inicio: 0}
    orden = []
    while frontera:
        estado = frontera.popleft()                 # FIFO: el mas antiguo
        orden.append(estado)
        if estado == meta:
            break
        for siguiente in sucesores(estado, paredes):
            if siguiente not in padre:              # no descubierto todavia
                padre[siguiente] = estado
                distancia[siguiente] = distancia[estado] + 1
                frontera.append(siguiente)
    return padre, distancia, orden


def reconstruir_camino(padre, meta):
    """Sigue los padres desde la meta hasta el inicio y lo da vuelta."""
    if meta not in padre:
        return None
    camino = []
    estado = meta
    while estado is not None:
        camino.append(estado)
        estado = padre[estado]
    return camino[::-1]


padre_bfs, distancia_bfs, orden_bfs = bfs(INICIO, META, PAREDES)
camino_bfs = reconstruir_camino(padre_bfs, META)
print(f"\n--- BFS ---")
print(f"Estados expandidos: {len(orden_bfs)} | descubiertos: {len(padre_bfs)}")
print(f"Largo del camino: {len(camino_bfs) - 1} pasos")
print(f"Camino: {camino_bfs}")


# ============================================================
# PARTE 4: DFS PARA COMPARAR
# ============================================================
# DFS (Depth-First Search, busqueda en profundidad) es el mismo codigo
# con una PILA (LIFO: el ultimo que entra sale primero) en vez de una
# cola. Se mete por un camino hasta el fondo antes de probar otro.

def dfs(inicio, meta, paredes):
    frontera = [inicio]                             # lista usada como pila
    padre = {inicio: None}
    orden = []
    while frontera:
        estado = frontera.pop()                     # LIFO: el mas nuevo
        orden.append(estado)
        if estado == meta:
            break
        for siguiente in sucesores(estado, paredes):
            if siguiente not in padre:
                padre[siguiente] = estado
                frontera.append(siguiente)
    return padre, orden


padre_dfs, orden_dfs = dfs(INICIO, META, PAREDES)
camino_dfs = reconstruir_camino(padre_dfs, META)
print(f"\n--- DFS ---")
print(f"Estados expandidos: {len(orden_dfs)}")
print(f"Largo del camino: {len(camino_dfs) - 1} pasos  (BFS: {len(camino_bfs) - 1})")


# ============================================================
# PARTE 5: UN AGENTE QUE BUSCA
# ============================================================
# El agente de limpieza de la 2.1, ahora basado en objetivos: conoce
# donde esta la suciedad y, cuando su celda esta limpia, usa BFS para
# ir por el camino mas corto a la celda sucia MAS CERCANA.
# BFS sirve para esto sin cambios: basta con frenar al encontrar el
# primer estado que cumpla la condicion "esta sucio".

def bfs_hasta(inicio, es_meta, paredes):
    frontera = deque([inicio])
    padre = {inicio: None}
    while frontera:
        estado = frontera.popleft()
        if es_meta(estado):
            return reconstruir_camino(padre, estado)
        for siguiente in sucesores(estado, paredes):
            if siguiente not in padre:
                padre[siguiente] = estado
                frontera.append(siguiente)
    return None


def limpiar_con_bfs(sucias, inicio, paredes):
    sucias = set(sucias)
    pos, pasos = inicio, 0
    while sucias:
        if pos in sucias:                                   # PERCIBIR + ACTUAR: limpiar
            sucias.remove(pos)
            pasos += 1
            continue
        camino = bfs_hasta(pos, lambda e: e in sucias, paredes)   # DECIDIR: buscar
        pos = camino[-1]                                    # ACTUAR: recorrer el camino
        pasos += len(camino) - 1
    return pasos


libres = [(f, c) for f in range(len(MAPA)) for c in range(len(MAPA[0])) if (f, c) not in PAREDES]
sucias = random.sample(libres, 8)
print(f"\n--- Agente de limpieza con BFS ---")
print(f"Celdas sucias: {sorted(sucias)}")
print(f"Pasos para limpiar las 8 celdas: {limpiar_con_bfs(sucias, INICIO, PAREDES)}")


# ============================================================
# PARTE 6: VISUALIZACION
# ============================================================
# (1) BFS: cada celda coloreada por su distancia al inicio (las "capas")
# (2) DFS: orden de exploracion
# (3) Tamano de cada capa de BFS

def matriz_mapa():
    return np.array([[1.0 if (f, c) in PAREDES else np.nan for c in range(len(MAPA[0]))] for f in range(len(MAPA))])


def dibujar_camino(ax, camino, color):
    ax.plot([c for _, c in camino], [f for f, _ in camino], color=color, linewidth=3)
    ax.text(INICIO[1], INICIO[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(META[1], META[0], "G", ha="center", va="center", fontweight="bold")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

capas = np.full((len(MAPA), len(MAPA[0])), np.nan)
for estado in orden_bfs:
    capas[estado] = distancia_bfs[estado]
ax1.imshow(matriz_mapa(), cmap="Greys", vmin=0, vmax=1.2)
im = ax1.imshow(capas, cmap="viridis")
fig.colorbar(im, ax=ax1, label="distancia al inicio (capa)")
dibujar_camino(ax1, camino_bfs, "#EF4444")
ax1.set_title(f"BFS: {len(orden_bfs)} expandidos, camino de {len(camino_bfs) - 1}")

orden_mat = np.full((len(MAPA), len(MAPA[0])), np.nan)
for i, estado in enumerate(orden_dfs):
    orden_mat[estado] = i
ax2.imshow(matriz_mapa(), cmap="Greys", vmin=0, vmax=1.2)
ax2.imshow(orden_mat, cmap="viridis")
dibujar_camino(ax2, camino_dfs, "#EF4444")
ax2.set_title(f"DFS: {len(orden_dfs)} expandidos, camino de {len(camino_dfs) - 1}")

tamano_capa = np.bincount([distancia_bfs[e] for e in orden_bfs])
ax3.bar(range(len(tamano_capa)), tamano_capa, color="#10B981")
ax3.set_title("Estados expandidos por capa (BFS)")
ax3.set_xlabel("Distancia al inicio")
ax3.set_ylabel("Estados")

for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("08_agent_search_bfs.png", dpi=120)
print("\nGrafico guardado en: 08_agent_search_bfs.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un problema de busqueda se formula como espacio de estados: estados,
   acciones (sucesores), estado inicial y estado meta
2. BFS usa una cola FIFO: explora por capas de distancia creciente
3. Por explorar en orden de distancia, BFS encuentra el camino con MENOS
   pasos (si todas las acciones cuestan lo mismo)
4. El diccionario padre evita repetir estados y permite reconstruir el
   camino desde la meta hacia atras
5. DFS es el mismo algoritmo con una pila LIFO: puede encontrar caminos
   mucho mas largos

EXPERIMENTOS SUGERIDOS:
- Agrega una pared que bloquee todos los caminos a G: que devuelve BFS?
- Cambia el orden de MOVIMIENTOS: cambia el largo del camino de BFS?
  y el de DFS?
- Quita paredes internas del mapa: como cambian los estados expandidos?
""")
