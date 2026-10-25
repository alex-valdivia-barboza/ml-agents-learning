"""
LECCION 2.3 - BUSQUEDA A*
=========================

Objetivo: buscar de forma INFORMADA. BFS (leccion 2.2) explora a ciegas,
en todas las direcciones por igual, y solo cuenta pasos. Aqui:
  1. Las acciones tienen COSTOS distintos: cruzar agua cuesta 5,
     caminar por tierra cuesta 1. El mejor camino es el mas barato,
     no el de menos pasos.
  2. El agente sabe donde esta la meta y usa esa informacion (una
     HEURISTICA) para explorar primero lo que parece mas prometedor.

A* ordena la frontera por  f(n) = g(n) + h(n):
  g(n) = costo real del camino desde el inicio hasta n
  h(n) = estimacion del costo que falta desde n hasta la meta
"""

import heapq

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PARTE 1: EL MAPA CON TERRENO
# ============================================================
# '#' = pared, '.' = tierra (costo 1), '~' = agua (costo 5)

MAPA = [
    "######################",
    "#....................#",
    "#....................#",
    "#.........~~~........#",
    "#....#....~~~........#",
    "#....#....~~~...#....#",
    "#S...#....~~~...#...G#",
    "#....#....~~~...#....#",
    "#.........~~~...#....#",
    "#.........~~~........#",
    "#.........~~~........#",
    "######################",
]
COSTO_TERRENO = {".": 1, "~": 5, "S": 1, "G": 1}


def leer_mapa(mapa):
    paredes, costo = set(), {}
    for f, fila in enumerate(mapa):
        for c, celda in enumerate(fila):
            if celda == "#":
                paredes.add((f, c))
            else:
                costo[(f, c)] = COSTO_TERRENO[celda]
                if celda == "S":
                    inicio = (f, c)
                elif celda == "G":
                    meta = (f, c)
    return paredes, costo, inicio, meta


PAREDES, COSTO, INICIO, META = leer_mapa(MAPA)
MOVIMIENTOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
print(f"Mapa {len(MAPA)}x{len(MAPA[0])} | inicio {INICIO} | meta {META}")


def sucesores(estado):
    """Devuelve (vecino, costo de entrar al vecino)."""
    f, c = estado
    return [((f + df, c + dc), COSTO[(f + df, c + dc)])
            for df, dc in MOVIMIENTOS if (f + df, c + dc) not in PAREDES]


# ============================================================
# PARTE 2: LA HEURISTICA
# ============================================================
# Distancia Manhattan: cuantos pasos harian falta si no hubiera paredes
# ni agua. Nunca SOBREESTIMA el costo real (cada paso cuesta al menos 1),
# por eso se dice que es ADMISIBLE.

def manhattan(estado, meta=META):
    return abs(estado[0] - meta[0]) + abs(estado[1] - meta[1])


print(f"h(inicio) = {manhattan(INICIO)}")


# ============================================================
# PARTE 3: UNA BUSQUEDA GENERICA CON COLA DE PRIORIDAD
# ============================================================
# La frontera es un heap (monticulo): heapq.heappop siempre saca el
# elemento con MENOR prioridad. Cambiando como se calcula la prioridad
# salen tres algoritmos distintos:
#   costo uniforme (Dijkstra): prioridad = g
#   busqueda golosa (greedy):  prioridad = h
#   A*:                        prioridad = g + h
# Con un peso w sobre h (g + w*h) se obtiene "A* ponderado".

def buscar(prioridad, inicio=INICIO, meta=META):
    contador = 0                                   # desempate: el orden de llegada
    frontera = [(prioridad(0, inicio), contador, inicio)]
    g = {inicio: 0}                                # mejor costo conocido hasta cada estado
    padre = {inicio: None}
    expandidos = []
    cerrados = set()
    while frontera:
        _, _, estado = heapq.heappop(frontera)
        if estado in cerrados:                     # ya expandido con un costo menor
            continue
        cerrados.add(estado)
        expandidos.append(estado)
        if estado == meta:
            break
        for vecino, costo_paso in sucesores(estado):
            g_nuevo = g[estado] + costo_paso
            if vecino not in g or g_nuevo < g[vecino]:     # camino nuevo o mas barato
                g[vecino] = g_nuevo
                padre[vecino] = estado
                contador += 1
                heapq.heappush(frontera, (prioridad(g_nuevo, vecino), contador, vecino))
    camino = []
    estado = meta
    while estado is not None:
        camino.append(estado)
        estado = padre.get(estado)
    return camino[::-1], g.get(meta), expandidos


def bfs(inicio=INICIO, meta=META):
    """BFS de la leccion 2.2: ignora los costos (minimiza pasos)."""
    from collections import deque
    frontera, padre, expandidos = deque([inicio]), {inicio: None}, []
    while frontera:
        estado = frontera.popleft()
        expandidos.append(estado)
        if estado == meta:
            break
        for vecino, _ in sucesores(estado):
            if vecino not in padre:
                padre[vecino] = estado
                frontera.append(vecino)
    camino, estado = [], meta
    while estado is not None:
        camino.append(estado)
        estado = padre[estado]
    camino = camino[::-1]
    return camino, sum(COSTO[e] for e in camino[1:]), expandidos


ALGORITMOS = {
    "BFS (solo pasos)": bfs,
    "Costo uniforme (g)": lambda: buscar(lambda g, n: g),
    "Golosa (h)": lambda: buscar(lambda g, n: manhattan(n)),
    "A* (g + h)": lambda: buscar(lambda g, n: g + manhattan(n)),
}

print("\n--- Comparacion ---")
resultados = {}
for nombre, algoritmo in ALGORITMOS.items():
    camino, costo, expandidos = algoritmo()
    resultados[nombre] = (camino, costo, expandidos)
    agua = sum(1 for e in camino if COSTO[e] == 5)
    print(f"{nombre:20s} | costo={costo:3d} | pasos={len(camino) - 1:3d} | "
          f"celdas de agua={agua} | expandidos={len(expandidos):3d}")


# ============================================================
# PARTE 4: HEURISTICA PONDERADA (experimento)
# ============================================================
# Multiplicar h por w > 1 la hace INADMISIBLE: puede sobreestimar.
# A* se vuelve mas "apurado": expande menos, pero ya no garantiza
# el camino mas barato.

print("\n--- A* ponderado: prioridad = g + w*h ---")
for w in [1, 1.5, 2, 5]:
    camino, costo, expandidos = buscar(lambda g, n, w=w: g + w * manhattan(n))
    print(f"w={w:<4} | costo={costo:3d} | expandidos={len(expandidos):3d}")


# ============================================================
# PARTE 5: VISUALIZACION
# ============================================================

fig, ejes = plt.subplots(1, 4, figsize=(20, 4.8))
base = np.array([[0.0 if ch == "#" else (0.35 if ch == "~" else 1.0) for ch in fila] for fila in MAPA])

for ax, (nombre, (camino, costo, expandidos)) in zip(ejes, resultados.items()):
    ax.imshow(base, cmap="Greys_r", vmin=-0.3, vmax=1.0)
    agua = np.array([[1.0 if ch == "~" else np.nan for ch in fila] for fila in MAPA])
    ax.imshow(agua, cmap="Blues", vmin=0, vmax=2, alpha=0.6)
    exp = np.full(base.shape, np.nan)
    for i, e in enumerate(expandidos):
        exp[e] = i
    ax.imshow(exp, cmap="viridis", alpha=0.75)
    ax.plot([c for _, c in camino], [f for f, _ in camino], color="#EF4444", linewidth=3)
    ax.set_title(f"{nombre}\ncosto {costo}, {len(expandidos)} expandidos")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("09_agent_search_astar.png", dpi=110)
print("\nGrafico guardado en: 09_agent_search_astar.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Con costos distintos, el camino mas corto en pasos no es el mas barato:
   BFS deja de ser optimo
2. Costo uniforme (Dijkstra) expande por menor g: optimo, pero ciego
3. Una heuristica h estima el costo restante. Manhattan es admisible:
   nunca sobreestima
4. A* expande por menor f = g + h: optimo si h es admisible, y explora
   menos que costo uniforme porque se orienta hacia la meta
5. La busqueda golosa (solo h) es rapida pero puede elegir caminos caros

EXPERIMENTOS SUGERIDOS:
- Cambia el costo del agua a 1: que algoritmos coinciden ahora?
- Cambia el costo del agua a 50: cambia el camino de A*?
- Usa h = 0 en A*: con que algoritmo coincide?
""")
