# INDICE MAESTRO — Redes Neuronales + Agentes desde Cero

Hoja de ruta completa del plan de aprendizaje. Dos modulos en paralelo,
2 horas diarias por modulo, pensados para 4-6 semanas.

- **Modulo 1**: Redes Neuronales, construidas con NumPy puro (sin frameworks tipo PyTorch/TensorFlow)
- **Modulo 2**: Agentes, construidos con Python puro

No hace falta terminar un modulo antes de empezar el otro: estan disenados
para avanzar en paralelo, alternando dias o sesiones.

## Tabla de lecciones

| # | Leccion | Archivo | Duracion | Modulo |
|---|---------|---------|----------|--------|
| 1.1 | Perceptron Simple | `01_perceptron_simple.py` | 2h | Redes |
| 2.1 | Agente Reactivo vs Deliberativo | `02_agent_hello_world.py` | 2h | Agentes |
| 1.2 | Red de 2 Capas | `03_neural_network_2layer.py` | 2h | Redes |
| 1.3 | Backpropagation Completa | `04_backpropagation_detailed.py` | 2h | Redes |
| 1.4 | Caso Real — MNIST | `05_mnist_classifier.py` | 2h | Redes |
| 1.5 | Funciones de Activacion | `06_activation_functions.py` | 1h | Redes |
| 1.6 | Regularizacion | `07_regularization.py` | 1.5h | Redes |
| 2.2 | Busqueda BFS | `08_agent_search_bfs.py` | 2h | Agentes |
| 2.3 | Busqueda A* | `09_agent_search_astar.py` | 2h | Agentes |
| 2.4 | Planificacion | `10_agent_planning.py` | 2h | Agentes |
| 2.5 | Q-Learning Basico | `11_agent_learning.py` | 2h | Agentes |
| 2.6 | Agentes con LLM (Claude API) | `12_agent_with_llm.py` | 2h | Agentes |

**Total estimado:** ~20.5 horas de contenido practico.

## Cronograma sugerido (4 semanas, 2h/dia entre ambos modulos)

| Semana | Dias | Contenido |
|--------|------|-----------|
| 1 | Dia 1-2 | 1.1 Perceptron Simple + 2.1 Agente Reactivo/Deliberativo |
| 1 | Dia 3-4 | 1.2 Red de 2 Capas + 2.2 Busqueda BFS |
| 1-2 | Dia 5-6 | 1.3 Backpropagation + 2.3 Busqueda A* |
| 2 | Dia 7-8 | 1.4 MNIST + 2.4 Planificacion |
| 2-3 | Dia 9 | 1.5 Funciones de Activacion (1h, dia mas liviano) |
| 3 | Dia 10 | 1.6 Regularizacion (1.5h) |
| 3 | Dia 11-12 | 2.5 Q-Learning Basico |
| 4 | Dia 13-14 | 2.6 Agentes con LLM + repaso general |

Si 2h diarias por modulo no es sostenible, priorizar terminar UN modulo
completo antes de arrancar el otro — es preferible profundidad a
paralelismo si el tiempo es limitado.

## Progresion de conceptos

**Modulo 1 (Redes):**
```
Neurona individual (1.1)
   -> Multiples neuronas en capas (1.2)
      -> Como se entrenan esas capas: backprop (1.3)
         -> Aplicacion en datos reales (1.4)
            -> Afinamiento: activaciones (1.5) y regularizacion (1.6)
```

**Modulo 2 (Agentes):**
```
Ciclo percibir-decidir-actuar (2.1)
   -> Explorar el espacio de estados sistematicamente: BFS (2.2)
      -> Explorar de forma inteligente con heuristicas: A* (2.3)
         -> Razonar sobre acciones y sus efectos: Planificacion (2.4)
            -> Aprender de la experiencia: Q-Learning (2.5)
               -> Delegar el razonamiento a un LLM (2.6)
```

**Punto de integracion:** las lecciones 2.5 (Q-Learning) y 1.x comparten
una idea central — ambas ajustan parametros numericos iterativamente
para reducir un error/maximizar una recompensa. Vale la pena notar el
paralelismo al llegar a esa leccion.

## Preguntas frecuentes

**¿Por que NumPy puro y no PyTorch/TensorFlow?**
El objetivo es entender QUE pasa matematicamente dentro de una red, no
solo usarla. Con un framework, backprop es una linea de codigo (`.backward()`)
que oculta el mecanismo. Escribiendolo a mano una vez, entender cualquier
framework despues es mucho mas facil.

**¿Necesito saber calculo para esto?**
Ayuda, pero no es obligatorio. Cada leccion introduce las derivadas que
necesita de forma minima y aplicada (ver `TEORIA_MINIMA.md`). No se
requiere calculo formal previo, solo estar dispuesto a seguir las
formulas paso a paso.

**¿En que orden hago las lecciones si tengo poco tiempo?**
El orden numerico de cada modulo (1.1 -> 1.2 -> ... y 2.1 -> 2.2 -> ...)
esta pensado para construir conceptos progresivamente. No conviene saltar
lecciones dentro de un mismo modulo.

**¿Que hago si un script tira un error?**
Revisar primero la seccion "Solucion de problemas" en `QUICK_START.md`.
Los errores mas comunes son de dependencias no instaladas (`numpy`,
`matplotlib`) o de version de Python.

**¿Donde anoto mi progreso?**
En `PROGRESO.md` — hay un checklist y espacio para notas por leccion.

**¿Que pasa con la leccion 2.6 (LLM) si no tengo API key de Claude?**
El archivo incluye un modo de fallback/simulacion que no requiere
conexion a la API, para poder seguir la logica del agente igual. Se
recomienda conseguir una API key en algun momento para ver el
comportamiento real, pero no es bloqueante para el resto del plan.
