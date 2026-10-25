"""
LECCION 2.6 - AGENTES CON LLM (CLAUDE API)
==========================================

Objetivo: delegar el paso de DECIDIR a un modelo de lenguaje.

En todo el modulo, el ciclo fue el mismo: PERCIBIR -> DECIDIR -> ACTUAR.
Lo que cambio fue como se decide: una regla (2.1), una busqueda (2.2,
2.3), un planificador (2.4), una tabla aprendida (2.5). Aqui decide
Claude, a partir de una instruccion en lenguaje natural.

El mecanismo se llama TOOL USE (uso de herramientas):
  1. Le damos a Claude una tarea y una lista de herramientas (funciones
     de Python que el puede pedir que ejecutemos).
  2. Claude responde pidiendo una herramienta ("mover hacia la derecha").
  3. Nosotros la ejecutamos en el entorno y le devolvemos el resultado.
  4. Se repite hasta que Claude da la tarea por terminada.

Dos modos:
  - REAL: si existe la variable de entorno ANTHROPIC_API_KEY (y el
    paquete anthropic >= 2.0: pip install -U anthropic). Cada llamada
    consume creditos de la API.
  - SIMULADO: sin clave, un "LLM de mentira" responde con el mismo
    formato que la API, para seguir el ciclo completo sin conexion.
    Para forzarlo aunque haya clave: MODO_SIMULADO=1
"""

import json
import os
from collections import deque

import matplotlib.pyplot as plt


# ============================================================
# PARTE 1: EL ENTORNO (la cuadricula de la 2.1, con una pared)
# ============================================================

MAPA = [
    "S*...",
    ".#*..",
    ".#...",
    ".*..*",
    "..*..",
]
DIRECCIONES = {"arriba": (-1, 0), "abajo": (1, 0), "izquierda": (0, -1), "derecha": (0, 1)}


class Entorno:
    def __init__(self, mapa):
        self.filas, self.columnas = len(mapa), len(mapa[0])
        self.paredes = {(f, c) for f in range(self.filas) for c in range(self.columnas) if mapa[f][c] == "#"}
        self.sucias = {(f, c) for f in range(self.filas) for c in range(self.columnas) if mapa[f][c] == "*"}
        self.pos = next((f, c) for f in range(self.filas) for c in range(self.columnas) if mapa[f][c] == "S")
        self.acciones = 0
        self.recorrido = [self.pos]

    def describir(self):
        """Texto que el agente recibe al percibir: el mapa completo."""
        lineas = []
        for f in range(self.filas):
            fila = ""
            for c in range(self.columnas):
                if (f, c) == self.pos:
                    fila += "@"
                elif (f, c) in self.paredes:
                    fila += "#"
                elif (f, c) in self.sucias:
                    fila += "*"
                else:
                    fila += "."
            lineas.append(fila)
        return ("\n".join(lineas) + f"\nPosicion (fila, columna): {self.pos}. "
                f"Celdas sucias restantes: {len(self.sucias)}. Leyenda: @ agente, # pared, * sucia.")

    def mover(self, direccion):
        if direccion not in DIRECCIONES:
            raise ValueError(f"direccion invalida: {direccion}")
        df, dc = DIRECCIONES[direccion]
        nueva = (self.pos[0] + df, self.pos[1] + dc)
        if not (0 <= nueva[0] < self.filas and 0 <= nueva[1] < self.columnas) or nueva in self.paredes:
            raise ValueError(f"no se puede mover {direccion} desde {self.pos}: hay pared o borde")
        self.pos = nueva
        self.acciones += 1
        self.recorrido.append(nueva)
        return f"Ahora en {self.pos}. {'La celda esta sucia.' if self.pos in self.sucias else 'La celda esta limpia.'}"

    def limpiar(self):
        self.acciones += 1
        if self.pos in self.sucias:
            self.sucias.remove(self.pos)
            return f"Celda {self.pos} limpia. Quedan {len(self.sucias)} sucias."
        return f"La celda {self.pos} ya estaba limpia (accion desperdiciada)."


# ============================================================
# PARTE 2: LAS HERRAMIENTAS (lo que Claude puede pedir)
# ============================================================
# Cada herramienta se describe con un nombre, una descripcion en
# lenguaje natural y un JSON Schema de sus parametros. Claude solo ve
# esto: no ve el codigo de Python. Una buena descripcion es parte del
# "prompt".

HERRAMIENTAS = [
    {
        "name": "ver_estado",
        "description": "Devuelve el mapa completo de la cuadricula, la posicion del agente y las celdas sucias. No gasta acciones.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "mover",
        "description": "Mueve el agente una celda. Gasta una accion. Falla si hay pared o borde.",
        "input_schema": {
            "type": "object",
            "properties": {"direccion": {"type": "string", "enum": list(DIRECCIONES)}},
            "required": ["direccion"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "limpiar",
        "description": "Limpia la celda donde esta el agente. Gasta una accion.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
]

SISTEMA = ("Controlas un robot de limpieza en una cuadricula. Usa las herramientas para "
           "limpiar todas las celdas sucias con la menor cantidad de acciones posible "
           "(cada mover y cada limpiar cuenta). Empieza mirando el estado. Planifica el "
           "recorrido antes de moverte y evita las paredes. Cuando no queden celdas sucias, "
           "responde con un resumen breve, sin llamar mas herramientas.")
TAREA = "Limpia toda la cuadricula."


def ejecutar_herramienta(entorno, nombre, entrada):
    """El 'ACTUAR': traduce el pedido de Claude a una accion en el entorno."""
    if nombre == "ver_estado":
        return entorno.describir()
    if nombre == "mover":
        return entorno.mover(entrada["direccion"])
    if nombre == "limpiar":
        return entorno.limpiar()
    raise ValueError(f"herramienta desconocida: {nombre}")


# ============================================================
# PARTE 3: EL CICLO DEL AGENTE (tool use)
# ============================================================
# Es el ciclo de la 2.1. La diferencia: DECIDIR = una llamada a la API.
# messages guarda toda la conversacion; la API no tiene memoria entre
# llamadas, asi que se reenvia completa cada vez.

def ciclo_agente(decidir, entorno, max_vueltas=60):
    mensajes = [{"role": "user", "content": TAREA}]
    llamadas, errores = 0, 0
    for _ in range(max_vueltas):
        respuesta = decidir(mensajes)                         # DECIDIR
        llamadas += 1
        mensajes.append({"role": "assistant", "content": respuesta.content})

        if respuesta.stop_reason == "refusal":
            print("El modelo rechazo la tarea.")
            break
        pedidos = [b for b in respuesta.content if b.type == "tool_use"]
        if not pedidos:                                       # termino: respondio con texto
            texto = next((b.text for b in respuesta.content if b.type == "text"), "")
            print(f"Respuesta final del modelo: {texto}")
            break

        resultados = []
        for pedido in pedidos:                                # ACTUAR (puede pedir varias a la vez)
            try:
                salida = ejecutar_herramienta(entorno, pedido.name, pedido.input)
                resultados.append({"type": "tool_result", "tool_use_id": pedido.id, "content": salida})
            except ValueError as error:                       # el error vuelve como informacion
                errores += 1
                resultados.append({"type": "tool_result", "tool_use_id": pedido.id,
                                   "content": f"Error: {error}", "is_error": True})
            print(f"  {pedido.name}({json.dumps(pedido.input, ensure_ascii=False)}) -> {resultados[-1]['content'].splitlines()[0]}")
        mensajes.append({"role": "user", "content": resultados})   # PERCIBIR: el resultado
    return llamadas, errores


# ============================================================
# PARTE 4: DOS FORMAS DE "DECIDIR"
# ============================================================

def decidir_con_claude():
    """Modo real. Requiere ANTHROPIC_API_KEY y anthropic >= 2.0."""
    import anthropic
    cliente = anthropic.Anthropic()                           # lee ANTHROPIC_API_KEY del entorno

    def decidir(mensajes):
        return cliente.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            system=SISTEMA,
            tools=HERRAMIENTAS,
            messages=mensajes,
            output_config={"effort": "low"},                  # tarea simple: poco razonamiento
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                              # si el modelo rechaza, reintenta con otro
        )
    return decidir


class Bloque:
    """Imita un bloque de contenido de la API (type, text, name, input, id)."""
    def __init__(self, **campos):
        self.__dict__.update(campos)


class Respuesta:
    """Imita la respuesta de la API: una lista de bloques y un stop_reason."""
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason


class LLMSimulado:
    """Un "LLM de mentira" para el modo sin conexion. Como un LLM real, solo
    sabe lo que aparece en la conversacion: lee el mapa y los resultados de
    las herramientas. Su estrategia: ir hacia la celda sucia mas cercana en
    linea recta (distancia Manhattan), sin planificar alrededor de las
    paredes. Si choca, prueba moverse en el otro eje."""

    def __init__(self):
        self.vueltas = 0
        self.pos, self.sucias, self.ultima_direccion = None, set(), None

    def leer_resultado(self, resultado):
        texto = resultado["content"]
        if "Leyenda" in texto:                                    # resultado de ver_estado
            lineas = texto.splitlines()[:-1]
            for f, fila in enumerate(lineas):
                for c, ch in enumerate(fila):
                    if ch == "*":
                        self.sucias.add((f, c))
                    elif ch == "@":
                        self.pos = (f, c)
        elif texto.startswith("Ahora en"):                        # resultado de mover
            dentro = texto[texto.index("(") + 1:texto.index(")")]
            self.pos = tuple(int(x) for x in dentro.split(","))
        elif texto.startswith("Celda"):                           # resultado de limpiar
            self.sucias.discard(self.pos)

    def __call__(self, mensajes):
        self.vueltas += 1
        bloque_id = f"toolu_sim_{self.vueltas}"
        ultimo = mensajes[-1]["content"]
        if isinstance(ultimo, str):                               # primer turno: mirar el estado
            return Respuesta([Bloque(type="tool_use", id=bloque_id, name="ver_estado", input={})], "tool_use")
        self.leer_resultado(ultimo[0])
        if not self.sucias:
            return Respuesta([Bloque(type="text", text="Listo: no quedan celdas sucias.")], "end_turn")
        if self.pos in self.sucias:
            return Respuesta([Bloque(type="tool_use", id=bloque_id, name="limpiar", input={})], "tool_use")
        objetivo = min(self.sucias, key=lambda s: abs(s[0] - self.pos[0]) + abs(s[1] - self.pos[1]))
        vertical = "abajo" if objetivo[0] > self.pos[0] else "arriba"
        horizontal = "derecha" if objetivo[1] > self.pos[1] else "izquierda"
        if ultimo[0].get("is_error"):                             # choco: cambiar de eje
            direccion = horizontal if self.ultima_direccion in ("arriba", "abajo") else vertical
        elif objetivo[0] != self.pos[0]:                          # primero filas, despues columnas
            direccion = vertical
        else:
            direccion = horizontal
        self.ultima_direccion = direccion
        return Respuesta([Bloque(type="tool_use", id=bloque_id, name="mover", input={"direccion": direccion})], "tool_use")


# ============================================================
# PARTE 5: EL AGENTE TRADICIONAL PARA COMPARAR (BFS de la 2.2)
# ============================================================

def agente_bfs(entorno):
    """Va siempre a la celda sucia mas cercana por el camino mas corto."""
    while entorno.sucias:
        if entorno.pos in entorno.sucias:
            entorno.limpiar()
            continue
        frontera, padre = deque([entorno.pos]), {entorno.pos: None}
        while frontera:
            e = frontera.popleft()
            if e in entorno.sucias:
                break
            for nombre, (df, dc) in DIRECCIONES.items():
                n = (e[0] + df, e[1] + dc)
                if 0 <= n[0] < entorno.filas and 0 <= n[1] < entorno.columnas and n not in entorno.paredes and n not in padre:
                    padre[n] = (e, nombre)
                    frontera.append(n)
        pasos = []
        while padre[e] is not None:
            e, nombre = padre[e]
            pasos.append(nombre)
        for nombre in reversed(pasos):
            entorno.mover(nombre)


# ============================================================
# PARTE 6: EJECUCION Y COMPARACION
# ============================================================

modo_real = bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("MODO_SIMULADO") != "1"
print(f"Modo: {'REAL (Claude Opus 5.5)' if modo_real else 'SIMULADO (sin conexion)'}\n")
print(Entorno(MAPA).describir(), "\n")

entorno_llm = Entorno(MAPA)
decidir = decidir_con_claude() if modo_real else LLMSimulado()
print("--- Agente con LLM ---")
llamadas, errores = ciclo_agente(decidir, entorno_llm)

entorno_bfs = Entorno(MAPA)
agente_bfs(entorno_bfs)

print("\n--- Comparacion ---")
print(f"Agente con LLM: {entorno_llm.acciones} acciones | {llamadas} llamadas al modelo | "
      f"{errores} errores de herramienta | sucias restantes: {len(entorno_llm.sucias)}")
print(f"Agente BFS:     {entorno_bfs.acciones} acciones | 0 llamadas al modelo | "
      f"sucias restantes: {len(entorno_bfs.sucias)}")


# ============================================================
# PARTE 7: VISUALIZACION
# ============================================================

fig, ejes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (titulo, ent) in zip(ejes, [("Agente con LLM" + (" (simulado)" if not modo_real else ""), entorno_llm),
                                    ("Agente BFS (2.2)", entorno_bfs)]):
    base = Entorno(MAPA)
    for (f, c) in base.paredes:
        ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#4B5563"))
    for (f, c) in base.sucias:
        ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#FDE68A"))
    ax.plot([c for _, c in ent.recorrido], [f for f, _ in ent.recorrido], color="#10B981", linewidth=3, marker="o")
    ax.set_xlim(-0.5, base.columnas - 0.5)
    ax.set_ylim(base.filas - 0.5, -0.5)
    ax.set_xticks(range(base.columnas))
    ax.set_yticks(range(base.filas))
    ax.grid(True, color="#E5E7EB")
    ax.set_title(f"{titulo}: {ent.acciones} acciones")

plt.tight_layout()
plt.savefig("12_agent_with_llm.png", dpi=120)
print("\nGrafico guardado en: 12_agent_with_llm.png")
print("Abriendo ventana con el grafico... (cerrala para que el script termine)")
plt.show()


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Un agente con LLM es el mismo ciclo percibir -> decidir -> actuar,
   con el "decidir" delegado a un modelo de lenguaje
2. Tool use: se describen herramientas (nombre, descripcion, JSON Schema);
   el modelo pide usarlas y el programa las ejecuta y devuelve el resultado
3. La API no tiene memoria: se reenvia la conversacion completa cada vez
4. Los errores de una herramienta se devuelven como informacion (is_error)
   para que el modelo corrija el rumbo
5. Un algoritmo clasico es mas barato y predecible si el problema esta bien
   definido; el LLM aporta flexibilidad para tareas en lenguaje natural

EXPERIMENTOS SUGERIDOS:
- Con clave de API: cambia TAREA por "Limpia solo la fila de abajo" sin
  tocar el codigo del agente. Que tendrias que reprogramar en el BFS?
- Quita la oracion "Planifica el recorrido antes de moverte" de SISTEMA
  y compara la cantidad de acciones
- Cambia la descripcion de la herramienta mover para que no diga que
  falla con paredes: aumentan los errores?
""")
