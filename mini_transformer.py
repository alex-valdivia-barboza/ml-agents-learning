"""
MINI TRANSFORMER CON mini_autograd (modulo 4)
=============================================

Las piezas de un transformer de solo decodificador (el tipo de GPT y de
Claude), escritas solo con el forward: los gradientes los calcula
mini_autograd. La leccion 4.2 prueba cada pieza; la 4.3 entrena con ellas
un mini-GPT.

Shapes: B = ejemplos del lote, T = posiciones, D = dimension del modelo,
H = cabezas, dh = D // H, V = vocabulario.
"""

import numpy as np

from mini_autograd import Tensor


def _pesos(rng, filas, columnas, escala=1.0):
    # misma idea que en la 1.4: dividir por raiz(entradas) para que la
    # salida tenga una escala parecida a la entrada
    return Tensor(rng.standard_normal((filas, columnas)) * escala / np.sqrt(filas))


# ---------- normalizacion ----------
def layer_norm(x, g, b, eps=1e-5):
    """Cada vector (ultimo eje) pasa a tener media 0 y varianza 1; despues
    g y b (aprendidos) le devuelven la escala y el corrimiento que convenga."""
    media = x.mean(axis=-1, keepdims=True)
    centrado = x - media
    varianza = (centrado ** 2).mean(axis=-1, keepdims=True)
    return centrado * (varianza + eps) ** -0.5 * g + b


# ---------- atencion con varias cabezas ----------
def mascara_causal(T):
    """-1e9 arriba de la diagonal: la posicion t no puede mirar t+1, t+2...
    Al pasar por la softmax, exp(-1e9) = 0."""
    return np.triu(np.full((T, T), -1e9), k=1)


def atencion(x, p, H, causal=True, guardar=None):
    B, T, D = x.shape
    dh = D // H

    def separar(t):                                    # (B, T, D) -> (B, H, T, dh)
        return t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    q, k, v = separar(x @ p["Wq"]), separar(x @ p["Wk"]), separar(x @ p["Wv"])
    puntajes = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)   # (B, H, T, T)  como en la 3.6
    if causal:
        puntajes = puntajes + mascara_causal(T)
    alfa = puntajes.softmax(axis=-1)
    if guardar is not None:
        guardar.append(alfa.data)
    salida = (alfa @ v).transpose(0, 2, 1, 3).reshape(B, T, D)   # juntar las cabezas
    return salida @ p["Wo"]


# ---------- red de cada posicion ----------
def mlp(x, p):
    return (x @ p["W1"] + p["b1"]).relu() @ p["W2"] + p["b2"]


# ---------- el bloque ----------
def init_bloque(rng, D, escala_salida=1.0):
    p = {n: _pesos(rng, D, D) for n in ["Wq", "Wk", "Wv"]}
    p["Wo"] = _pesos(rng, D, D, escala_salida)
    p["W1"], p["b1"] = _pesos(rng, D, 4 * D), Tensor(np.zeros(4 * D))
    p["W2"], p["b2"] = _pesos(rng, 4 * D, D, escala_salida), Tensor(np.zeros(D))
    for n in ["ln1", "ln2"]:
        p[n + "_g"], p[n + "_b"] = Tensor(np.ones(D)), Tensor(np.zeros(D))
    return p


def bloque(x, p, H, causal=True, residual=True, norma=True, guardar=None):
    """x = x + atencion(norma(x));  x = x + mlp(norma(x))"""
    n1 = layer_norm(x, p["ln1_g"], p["ln1_b"]) if norma else x
    a = atencion(n1, p, H, causal, guardar)
    x = x + a if residual else a
    n2 = layer_norm(x, p["ln2_g"], p["ln2_b"]) if norma else x
    m = mlp(n2, p)
    return x + m if residual else m


# ---------- el modelo completo ----------
def init_modelo(V, D, H, n_bloques, T_max, posiciones=True, semilla=0):
    rng = np.random.default_rng(semilla)
    modelo = {"V": V, "D": D, "H": H, "posiciones": posiciones,
              "E": Tensor(rng.standard_normal((V, D)) * 0.5),           # embedding de cada token
              "P": Tensor(rng.standard_normal((T_max, D)) * 0.5),       # embedding de cada posicion
              "bloques": [init_bloque(rng, D, escala_salida=1 / np.sqrt(2 * n_bloques))
                          for _ in range(n_bloques)],
              "lnf_g": Tensor(np.ones(D)), "lnf_b": Tensor(np.zeros(D)),
              "W_salida": _pesos(rng, D, V)}
    return modelo


def parametros(modelo):
    lista = [modelo["E"], modelo["lnf_g"], modelo["lnf_b"], modelo["W_salida"]]
    if modelo["posiciones"]:
        lista.append(modelo["P"])
    for p in modelo["bloques"]:
        lista.extend(p.values())
    return lista


def contar(modelo):
    return sum(p.data.size for p in parametros(modelo))


def forward(modelo, tokens, guardar=None, **opciones):
    """tokens: array de enteros (B, T) -> logits (B, T, V)"""
    B, T = tokens.shape
    x = modelo["E"][tokens]                                         # (B, T, D)  como en la 3.4
    if modelo["posiciones"]:
        x = x + modelo["P"][np.arange(T)]                           # (T, D) se suma a cada ejemplo
    for p in modelo["bloques"]:
        x = bloque(x, p, modelo["H"], guardar=guardar, **opciones)
    x = layer_norm(x, modelo["lnf_g"], modelo["lnf_b"])
    return x @ modelo["W_salida"]


# ---------- guardar y cargar (la 4.5 usa el modelo entrenado en la 4.3) ----------
def guardar(modelo, ruta, **extra):
    arrays = {f"p{i}": p.data for i, p in enumerate(parametros(modelo))}
    config = [modelo["V"], modelo["D"], modelo["H"], len(modelo["bloques"]),
              modelo["P"].shape[0], int(modelo["posiciones"])]
    np.savez(ruta, config=np.array(config), **arrays, **extra)


def cargar(ruta):
    archivo = np.load(ruta, allow_pickle=False)
    V, D, H, n_bloques, T_max, posiciones = archivo["config"]
    modelo = init_modelo(V, D, H, n_bloques, T_max, bool(posiciones))
    for i, p in enumerate(parametros(modelo)):
        p.data = archivo[f"p{i}"]
        p.grad = np.zeros_like(p.data)
    return modelo, archivo
