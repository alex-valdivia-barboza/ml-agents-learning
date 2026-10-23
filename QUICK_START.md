# QUICK START — Primeros 15 minutos

Guia rapida para arrancar sin trabas. Si ya tenes Python, numpy y
matplotlib instalados, anda directo al paso 3.

## Paso 1: Verificar Python

Abri una terminal (PowerShell) en esta carpeta y corre:

```
python --version
```

Necesitas Python 3.7 o superior. Si el comando no se reconoce, instala
Python desde https://www.python.org/downloads/ (marca la opcion
"Add python.exe to PATH" durante la instalacion).

## Paso 2: Instalar dependencias

Este proyecto usa NumPy (calculo numerico) y matplotlib (graficos).
No se usan frameworks de deep learning (PyTorch/TensorFlow) a
proposito — el objetivo es entender el mecanismo interno.

```
python -m pip install numpy matplotlib
```

Verificar que quedaron instaladas:

```
python -c "import numpy, matplotlib; print('OK')"
```

Si ves `OK`, estas listo.

## Paso 3: Ejecutar la primera leccion

```
python 01_perceptron_simple.py
```

Deberias ver en la consola: datos generados, el progreso del
entrenamiento epoca por epoca, y al final un mensaje confirmando que
se guardo un grafico `01_perceptron_simple.png` en esta misma carpeta.
Ademas se abre una ventana con el grafico (datos + frontera de
decision aprendida) — cerrala para que el script termine de imprimir
los conceptos clave.

## Paso 4: Ejecutar la segunda leccion

```
python 02_agent_hello_world.py
```

Vas a ver dos simulaciones (agente reactivo y agente deliberativo)
limpiando una cuadricula, con una comparacion de cuantos pasos usó
cada uno al final.

## Paso 5: Seguir el orden del indice

A partir de aca, segui el orden sugerido en `00_INDEX_MASTER.md`.
Cada leccion es un archivo `.py` que se ejecuta igual que los dos
anteriores: `python NN_nombre_leccion.py`.

Antes de cada leccion nueva, lee primero el docstring al inicio del
archivo (el bloque de texto entre `"""..."""`) — explica el objetivo
y los conceptos antes de mostrar el codigo.

## Solucion de problemas

**`'python' no se reconoce como un comando...`**
Python no esta en el PATH. Reinstala Python marcando "Add to PATH", o
usa `py` en vez de `python` (a veces el instalador de Windows solo
registra ese alias).

**`ModuleNotFoundError: No module named 'numpy'` (o `matplotlib`)**
No se instalaron las dependencias. Repeti el Paso 2. Si tenes varias
instalaciones de Python, asegurate de instalar con el mismo `python`
que usas para correr los scripts (`python -m pip install ...` en vez
de solo `pip install ...` evita ese problema).

**El grafico no se abre en una ventana**
Los scripts GUARDAN los graficos como archivos `.png` en la carpeta y
ademas intentan abrir una ventana interactiva (`plt.show()`). Si la
ventana no aparece, puede ser que falte el backend grafico de
matplotlib (Tkinter) — en Windows normalmente ya viene con Python. De
todas formas el `.png` se genera igual, asi que siempre podes abrirlo
manualmente con cualquier visor de imagenes.

**El script queda "colgado" despues de entrenar**
No esta colgado: esta esperando a que cierres la ventana del grafico
(`plt.show()` pausa la ejecucion hasta que la cierres). Cerra la
ventana para que el script continue e imprima los conceptos clave.

**Los caracteres con tildes/eñes se ven raros en la consola**
Es un tema de codificacion de la terminal de Windows (cp1252 vs
UTF-8), no un error del script. El contenido del archivo esta bien;
si molesta, corre `chcp 65001` antes de ejecutar los scripts para
cambiar la consola a UTF-8.

**Un script tarda mucho o parece colgado**
Las lecciones que entrenan sobre datasets mas grandes (como
`05_mnist_classifier.py`) pueden tardar uno o varios minutos segun la
computadora — es esperable, no es un error. El script imprime
progreso epoca por epoca para confirmar que sigue corriendo.

**Quiero reiniciar una leccion desde cero**
Los scripts no modifican datos externos ni tienen estado persistente
(salvo los `.png` que generan, que se sobreescriben en cada corrida).
Simplemente volve a ejecutar `python NN_archivo.py`.

## Como registrar tu progreso

Cada vez que termines una leccion, marcala en `PROGRESO.md` y anota
cualquier duda o insight en la seccion de notas de esa leccion.
