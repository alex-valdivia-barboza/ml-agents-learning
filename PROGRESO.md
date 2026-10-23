# PROGRESO — Checklist y notas de aprendizaje

Marca cada leccion como completada cuando: (1) ejecutaste el script sin
errores, (2) leiste y entendiste los "Conceptos clave" al final, y
(3) probaste al menos uno de los "Experimentos sugeridos".

## Checklist

### Modulo 1 — Redes Neuronales

- [ ] **1.1 Perceptron Simple** (`01_perceptron_simple.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo que es forward pass (z = X.w + b, y = sigma(z))
  - [ ] Entiendo como se actualizan los pesos al entrenar
  - [ ] Probe al menos un experimento sugerido
- [ ] **1.2 Red de 2 Capas** (`03_neural_network_2layer.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo por que una capa oculta permite resolver XOR
  - [ ] Probe al menos un experimento sugerido
- [ ] **1.3 Backpropagation Completa** (`04_backpropagation_detailed.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo como el error se propaga hacia atras entre capas
  - [ ] Probe al menos un experimento sugerido
- [ ] **1.4 Caso Real — MNIST** (`05_mnist_classifier.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo la metrica accuracy y por que se mide en datos separados
  - [ ] Probe al menos un experimento sugerido
- [ ] **1.5 Funciones de Activacion** (`06_activation_functions.py`)
  - [ ] Ejecutado sin errores
  - [ ] Puedo explicar cuando usar sigmoid vs ReLU vs tanh
- [ ] **1.6 Regularizacion** (`07_regularization.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo que es overfitting y como L1/L2/Dropout lo mitigan

### Modulo 2 — Agentes

- [ ] **2.1 Agente Reactivo vs Deliberativo** (`02_agent_hello_world.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo el ciclo percibir -> decidir -> actuar
  - [ ] Entiendo la diferencia entre agente con y sin memoria
  - [ ] Probe al menos un experimento sugerido
- [ ] **2.2 Busqueda BFS** (`08_agent_search_bfs.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo por que BFS garantiza el camino mas corto
  - [ ] Probe al menos un experimento sugerido
- [ ] **2.3 Busqueda A*** (`09_agent_search_astar.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo que es una heuristica admisible
  - [ ] Comparé cuantos nodos explora A* vs BFS
- [ ] **2.4 Planificacion** (`10_agent_planning.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo precondiciones y efectos de una accion
- [ ] **2.5 Q-Learning Basico** (`11_agent_learning.py`)
  - [ ] Ejecutado sin errores
  - [ ] Entiendo la formula de actualizacion de Q
  - [ ] Entiendo exploracion vs explotacion (epsilon-greedy)
- [ ] **2.6 Agentes con LLM (Claude API)** (`12_agent_with_llm.py`)
  - [ ] Ejecutado sin errores (con o sin API key)
  - [ ] Comparé el enfoque tradicional vs el basado en LLM

## Validacion de comprension (autoevaluacion)

Despues de cada modulo, intenta responder sin mirar el codigo:

**Modulo 1:**
1. ¿Que hace la funcion de activacion en una neurona, y por que hace
   falta que sea NO lineal?
2. ¿Que mide la funcion de perdida y como se usa para actualizar pesos?
3. ¿Por que backpropagation empieza en la salida y va hacia atras?

**Modulo 2:**
1. ¿Cual es la diferencia entre un agente reactivo y uno deliberativo?
2. ¿Cuando conviene usar A* en vez de BFS?
3. ¿En que se diferencia Q-Learning de un plan generado por STRIPS?

## Notas de aprendizaje

_(Usa este espacio para anotar dudas, insights, o cosas que te costo
entender en cada leccion. Revisitar estas notas antes de arrancar el
siguiente modulo ayuda a fijar los conceptos.)_

### Modulo 1
- Leccion 1.1:
- Leccion 1.2:
- Leccion 1.3:
- Leccion 1.4:
- Leccion 1.5:
- Leccion 1.6:

### Modulo 2
- Leccion 2.1:
- Leccion 2.2:
- Leccion 2.3:
- Leccion 2.4:
- Leccion 2.5:
- Leccion 2.6:

## Ciclo diario sugerido

1. Elegir la siguiente leccion pendiente en el checklist (una de cada
   modulo si el tiempo lo permite)
2. Leer el docstring inicial del archivo antes de correrlo
3. Ejecutar el script y observar el output paso a paso
4. Abrir el/los `.png` generados y analizar la visualizacion
5. Leer la seccion "Conceptos clave" al final del script
6. Probar al menos un "Experimento sugerido", modificando el codigo
7. Marcar la leccion como completada aca y anotar notas si aplica
