"""
LECCION 2.1 - AGENTE REACTIVO VS DELIBERATIVO
===============================================

Objetivo: entender el ciclo basico de un agente: PERCIBIR -> DECIDIR -> ACTUAR

Entorno: una cuadricula (grid) de 3x3 donde cada celda puede estar
"sucia" o "limpia". El agente se mueve por la cuadricula y limpia
las celdas sucias.

Vamos a comparar dos tipos de agente:
  - Agente REACTIVO: decide solo con lo que percibe AHORA, sin memoria.
  - Agente DELIBERATIVO: recuerda que celdas ya visito, y usa esa
    memoria para decidir mejor a donde ir.
"""

import random

random.seed(42)


# ============================================================
# PARTE 1: EL ENTORNO
# ============================================================

class Entorno:
    """Cuadricula de 3x3. Cada celda es True si esta sucia."""

    def __init__(self, tamano=3, prob_sucia=0.5):
        self.tamano = tamano
        # grid[fila][columna] = True (sucia) / False (limpia)
        self.grid = [
            [random.random() < prob_sucia for _ in range(tamano)]
            for _ in range(tamano)
        ]

    def esta_sucia(self, pos):
        fila, col = pos
        return self.grid[fila][col]

    def limpiar(self, pos):
        fila, col = pos
        self.grid[fila][col] = False

    def todo_limpio(self):
        return not any(any(fila) for fila in self.grid)

    def celdas_sucias_restantes(self):
        return sum(sum(fila) for fila in self.grid)

    def imprimir(self, pos_agente):
        """Muestra la cuadricula en texto: 'S'=sucia, '.'=limpia, '@'=agente"""
        for f in range(self.tamano):
            fila_str = ""
            for c in range(self.tamano):
                if (f, c) == pos_agente:
                    fila_str += "@ "
                elif self.grid[f][c]:
                    fila_str += "S "
                else:
                    fila_str += ". "
            print(fila_str)
        print()


# ============================================================
# PARTE 2: AGENTE REACTIVO (sin memoria)
# ============================================================
# Ciclo: PERCIBIR la celda actual -> si esta sucia, LIMPIAR.
#        Si esta limpia, decide moverse con una regla simple de
#        estimulo-respuesta (sin recordar donde ya estuvo): elige
#        una celda vecina al azar. Es el modelo clasico de un
#        robot aspiradora reactivo tipo "rebote aleatorio".
#        Como no tiene memoria, puede volver a pasar por celdas
#        que ya visito antes -> es menos eficiente.

class AgenteReactivo:
    nombre = "Reactivo"

    def __init__(self, tamano):
        self.tamano = tamano
        self.pos = (0, 0)

    def celdas_vecinas(self, pos):
        fila, col = pos
        vecinas = []
        for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nf, nc = fila + df, col + dc
            if 0 <= nf < self.tamano and 0 <= nc < self.tamano:
                vecinas.append((nf, nc))
        return vecinas

    def percibir_y_actuar(self, entorno):
        """Un paso del ciclo percibir -> decidir -> actuar."""
        # PERCIBIR: solo mira si la celda actual esta sucia
        sucia = entorno.esta_sucia(self.pos)

        if sucia:
            # ACTUAR: limpiar
            entorno.limpiar(self.pos)
            return "limpiar"
        else:
            # DECIDIR (regla fija, sin memoria): moverse a una celda
            # vecina al azar. No sabe si ya visito esa celda antes,
            # por eso puede quedar dando vueltas por el mismo lugar.
            vecinas = self.celdas_vecinas(self.pos)
            self.pos = random.choice(vecinas)
            return f"mover a {self.pos}"


# ============================================================
# PARTE 3: AGENTE DELIBERATIVO (con memoria)
# ============================================================
# Recuerda que celdas ya visito. Al decidir a donde moverse,
# prioriza celdas NO visitadas en vez de repetir camino.

class AgenteDeliberativo:
    nombre = "Deliberativo"

    def __init__(self, tamano):
        self.tamano = tamano
        self.pos = (0, 0)
        self.visitadas = set()
        self.visitadas.add(self.pos)

    def celdas_vecinas(self, pos):
        fila, col = pos
        vecinas = []
        for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nf, nc = fila + df, col + dc
            if 0 <= nf < self.tamano and 0 <= nc < self.tamano:
                vecinas.append((nf, nc))
        return vecinas

    def percibir_y_actuar(self, entorno):
        sucia = entorno.esta_sucia(self.pos)

        if sucia:
            entorno.limpiar(self.pos)
            return "limpiar"
        else:
            # DECIDIR con memoria: preferir una celda vecina que
            # todavia NO haya sido visitada
            vecinas = self.celdas_vecinas(self.pos)
            no_visitadas = [c for c in vecinas if c not in self.visitadas]

            if no_visitadas:
                self.pos = no_visitadas[0]
            else:
                # si ya visito todas las vecinas, se mueve a cualquiera
                # (evita quedarse trabado)
                self.pos = vecinas[0]

            self.visitadas.add(self.pos)
            return f"mover a {self.pos}"


# ============================================================
# PARTE 4: SIMULACION
# ============================================================

def simular(agente_clase, tamano=3, prob_sucia=0.5, max_pasos=50, mostrar=True):
    random.seed(7)  # mismo entorno para comparar ambos agentes de forma justa
    entorno = Entorno(tamano=tamano, prob_sucia=prob_sucia)
    agente = agente_clase(tamano)

    if mostrar:
        print(f"=== Agente {agente.nombre} - estado inicial ===")
        entorno.imprimir(agente.pos)

    pasos = 0
    while not entorno.todo_limpio() and pasos < max_pasos:
        accion = agente.percibir_y_actuar(entorno)
        pasos += 1
        if mostrar and pasos <= 6:
            print(f"Paso {pasos}: accion={accion} | celdas sucias restantes={entorno.celdas_sucias_restantes()}")

    if mostrar:
        print(f"\n=== Agente {agente.nombre} - estado final (paso {pasos}) ===")
        entorno.imprimir(agente.pos)
        print(f"Total de pasos usados: {pasos}")
        print(f"Cuadricula limpia por completo: {entorno.todo_limpio()}\n")

    return pasos


print("############################################")
print("# SIMULACION 1: AGENTE REACTIVO")
print("############################################\n")
pasos_reactivo = simular(AgenteReactivo)

print("############################################")
print("# SIMULACION 2: AGENTE DELIBERATIVO")
print("############################################\n")
pasos_deliberativo = simular(AgenteDeliberativo)


# ============================================================
# PARTE 5: COMPARACION DE EFICIENCIA
# ============================================================

print("############################################")
print("# COMPARACION DE EFICIENCIA")
print("############################################")
print(f"Agente Reactivo:     {pasos_reactivo} pasos para limpiar todo")
print(f"Agente Deliberativo: {pasos_deliberativo} pasos para limpiar todo")

if pasos_deliberativo < pasos_reactivo:
    print("-> El agente con memoria fue mas eficiente: evito repetir celdas ya visitadas.")
elif pasos_deliberativo > pasos_reactivo:
    print("-> En esta corrida el reactivo fue mas rapido (puede pasar con grids chicos).")
else:
    print("-> Ambos usaron la misma cantidad de pasos en esta corrida.")


# ============================================================
# CONCEPTOS CLAVE
# ============================================================
print("""
CONCEPTOS CLAVE DE ESTA LECCION:
1. Ciclo del agente: PERCIBIR (leer el entorno) -> DECIDIR (elegir accion)
   -> ACTUAR (ejecutarla y modificar el entorno)
2. Agente REACTIVO: decide solo con la percepcion actual. Simple y rapido
   de implementar, pero puede repetir trabajo (no sabe donde ya estuvo).
3. Agente DELIBERATIVO: mantiene un ESTADO INTERNO (memoria) que usa
   para tomar mejores decisiones. Mas complejo, pero mas eficiente.
4. Este es el primer paso hacia agentes mas sofisticados: en las
   proximas lecciones el agente va a "pensar antes de actuar" usando
   busqueda (BFS, A*) y planificacion, en vez de reglas fijas.

EXPERIMENTOS SUGERIDOS:
- Cambia `tamano=5` y `prob_sucia=0.7`: como cambia la diferencia
  de eficiencia entre ambos agentes a mayor escala?
- Modifica AgenteReactivo para que se mueva en linea recta (derecha,
  y al llegar al borde baja una fila) en vez de al azar: por que
  este agente reactivo con una regla fija resulta mas eficiente que
  el que se mueve al azar, aun sin tener memoria?
- Agrega un contador de "movimientos desperdiciados" (moverse a una
  celda ya visitada) para cada agente y compara ese numero.
""")
