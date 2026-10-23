# TEORIA MINIMA — La matematica esencial, sin exceso

Este documento junta en un solo lugar las formulas y conceptos que
usan los scripts del proyecto. La idea NO es reemplazar un curso de
calculo o algebra lineal, sino dar justo lo necesario para seguir el
codigo con entendimiento real de por que funciona.

---

## SECCION 1: Redes Neuronales

### 1.1 La neurona

Una neurona artificial toma un vector de entradas `X`, las combina
linealmente con pesos `w` y un sesgo `b`, y aplica una funcion de
activacion:

```
z = X . w + b        (combinacion lineal — "que tan fuerte es la señal")
a = activacion(z)     (no-linealidad — "que tanto se dispara la neurona")
```

`X . w` es un producto punto: multiplica cada entrada por su peso
correspondiente y suma todo. Si `X = [x1, x2]` y `w = [w1, w2]`:

```
X . w = x1*w1 + x2*w2
```

### 1.2 Funciones de activacion

**Sigmoid** — aplasta cualquier numero real al rango (0, 1):
```
sigmoid(z) = 1 / (1 + e^-z)
```
Util para representar probabilidades. Se usa en `01_perceptron_simple.py`.

**ReLU** (Rectified Linear Unit) — la mas usada en redes modernas:
```
relu(z) = max(0, z)
```
Simple y evita que el gradiente se "desvanezca" en valores positivos.

**Tanh** — como sigmoid pero en el rango (-1, 1):
```
tanh(z) = (e^z - e^-z) / (e^z + e^-z)
```

Ver `06_activation_functions.py` para comparar las tres visualmente.

### 1.3 Funcion de perdida (loss / error)

Mide que tan lejos esta la prediccion `y_pred` del valor real `y`.
La mas simple es el error cuadratico medio (MSE):

```
error = y - y_pred
MSE = promedio(error^2)
```

Entrenar una red = encontrar los pesos que MINIMIZAN esta funcion de
perdida.

### 1.4 Gradiente y descenso de gradiente

El gradiente de la perdida respecto a un peso nos dice: "si aumento
este peso un poquito, ¿la perdida sube o baja, y cuanto?"

Descenso de gradiente = mover el peso en la direccion CONTRARIA al
gradiente (para reducir la perdida), un paso pequeño a la vez:

```
peso_nuevo = peso_viejo - tasa_aprendizaje * gradiente
```

La `tasa_aprendizaje` (learning rate) controla el tamaño del paso.
Muy alta -> el entrenamiento se vuelve inestable. Muy baja -> aprende
demasiado lento.

### 1.5 Regla de la cadena y backpropagation

Cuando hay varias capas, el error se calcula al final (en la salida),
pero necesitamos saber cuanto contribuyo CADA peso de CADA capa a ese
error. La regla de la cadena del calculo permite "propagar" el error
hacia atras, capa por capa:

```
d(error)/d(peso_capa_1) = d(error)/d(salida) * d(salida)/d(capa_2) * d(capa_2)/d(peso_capa_1)
```

En la practica, esto se calcula de atras hacia adelante (de la salida
hacia la entrada) — por eso se llama BACKpropagation. Cada capa solo
necesita conocer el "error" que le llega de la capa siguiente (llamado
`delta`) para calcular sus propios gradientes.

Ver `04_backpropagation_detailed.py` para el desarrollo completo,
capa por capa.

### 1.6 Regularizacion (evitar overfitting)

Overfitting = la red memoriza los datos de entrenamiento en vez de
aprender un patron general, y por eso funciona mal con datos nuevos.

- **L2 (Ridge):** penaliza pesos grandes, sumando `lambda * suma(w^2)`
  a la funcion de perdida. Empuja los pesos hacia valores pequeños.
- **L1:** parecido, pero con `lambda * suma(|w|)`. Tiende a llevar
  algunos pesos exactamente a cero (selecciona features).
- **Dropout:** durante el entrenamiento, apaga neuronas al azar en
  cada iteracion. Evita que la red dependa demasiado de una sola
  neurona especifica.

Ver `07_regularization.py`.

---

## SECCION 2: Agentes

### 2.1 El ciclo del agente

Todo agente, sin importar su complejidad, sigue el mismo ciclo:

```
PERCIBIR (leer el estado del entorno)
   -> DECIDIR (elegir una accion, con alguna logica/algoritmo)
      -> ACTUAR (ejecutar la accion, que modifica el entorno)
         -> (repetir)
```

Lo que cambia entre un agente simple y uno sofisticado es que pasa
en el paso "DECIDIR": una regla fija, memoria, busqueda, aprendizaje, etc.

### 2.2 Espacio de estados

Para que un agente pueda "buscar" una solucion, el problema se modela
como un **espacio de estados**:

- **Estado:** una configuracion posible del problema (ej: posicion
  del agente en una grilla)
- **Acciones:** formas de pasar de un estado a otro
- **Estado inicial** y **estado meta** (objetivo)

Buscar una solucion = encontrar un camino desde el estado inicial
hasta el estado meta, a traves de acciones validas.

### 2.3 Busqueda BFS (Breadth-First Search)

Explora el espacio de estados "por capas": primero todos los estados
a 1 paso del inicio, despues todos los que estan a 2 pasos, etc.

Usa una **cola FIFO** (first in, first out) para decidir que estado
explorar despues, y un conjunto de **visitados** para no repetir
trabajo.

**Garantia importante:** si todas las acciones tienen el mismo costo,
BFS siempre encuentra el camino MAS CORTO (optimo).

Ver `08_agent_search_bfs.py`.

### 2.4 Busqueda A*

Mejora BFS usando una **heuristica**: una estimacion de "que tan
lejos estoy de la meta" desde cada estado. En vez de explorar a
ciegas, A* prioriza los estados que PARECEN estar mas cerca de la
meta.

```
f(estado) = g(estado) + h(estado)

g(estado) = costo real acumulado desde el inicio hasta este estado
h(estado) = estimacion heuristica del costo restante hasta la meta
```

A* explora primero los estados con menor `f`. Si la heuristica es
**admisible** (nunca sobreestima el costo real restante), A* tambien
garantiza encontrar el camino optimo, pero tipicamente explorando
MENOS estados que BFS.

Ver `09_agent_search_astar.py`.

### 2.5 Planificacion (STRIPS)

En vez de explorar estado por estado, un planificador razona sobre
**acciones simbolicas**, cada una definida por:

- **Precondiciones:** que debe ser verdad para poder ejecutar la accion
- **Efectos:** que cambia en el mundo al ejecutarla

Ejemplo simplificado (mover un bloque):
```
Accion: mover(bloque, de, a)
Precondicion: bloque esta en "de", "a" esta libre
Efecto: bloque ahora esta en "a", "de" queda libre
```

Un plan es una SECUENCIA de acciones que lleva del estado inicial al
estado meta, cumpliendo las precondiciones de cada paso.

Ver `10_agent_planning.py`.

### 2.6 Q-Learning (aprendizaje por refuerzo)

El agente aprende, por prueba y error, el VALOR de cada accion en
cada estado, guardado en una **tabla Q**: `Q[estado][accion]`.

Formula de actualizacion:
```
Q[s][a] = Q[s][a] + alpha * (r + gamma * max(Q[s'][a']) - Q[s][a])
```

- `alpha` (tasa de aprendizaje): cuanto confiar en la nueva informacion
- `r`: recompensa (reward) inmediata obtenida
- `gamma` (factor de descuento): cuanto importa la recompensa futura
  vs la inmediata
- `max(Q[s'][a'])`: la mejor accion posible desde el proximo estado

**Exploracion vs explotacion (ε-greedy):** el agente a veces elige la
mejor accion conocida (explotar) y a veces una accion al azar
(explorar), para descubrir posibles mejores opciones que aun no
conoce. `epsilon` controla la probabilidad de explorar.

Ver `11_agent_learning.py`.

---

## SECCION 3: Integracion (donde se conectan ambos modulos)

Redes neuronales y agentes no son temas separados — en sistemas
modernos (como Deep Q-Learning o agentes basados en LLM) se combinan:

- **Q-Learning con tabla** (leccion 2.5) funciona cuando el numero de
  estados es chico. Cuando el espacio de estados es enorme (ej: pixeles
  de un videojuego), se reemplaza la tabla Q por una RED NEURONAL que
  APROXIMA los valores Q — eso se llama Deep Q-Learning (DQN), y es la
  evolucion natural de unir ambos modulos.
- **Agentes con LLM** (leccion 2.6) reemplazan la logica de "decidir"
  del ciclo del agente por una red neuronal enorme ya entrenada (el
  LLM), a la que se le da contexto en forma de texto (prompt) en vez
  de programar reglas o tablas a mano.

La progresion pedagogica de este plan es exactamente esa: entender
primero los mecanismos basicos por separado (una neurona, un ciclo de
agente), para despues poder entender COMO y POR QUE se combinan en
sistemas mas avanzados.
