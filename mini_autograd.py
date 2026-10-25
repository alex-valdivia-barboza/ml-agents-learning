"""
MINI AUTOGRAD CON NUMPY (modulo 4)
==================================

Un Tensor guarda un array de NumPy (data), su gradiente (grad) y COMO se
calculo: de que tensores salio (_previos) y que funcion pasa el gradiente
hacia ellos (_atras). Con eso, backward() recorre el grafo al reves y
aplica la regla de la cadena automaticamente.

Es la version con arrays de la clase Valor de 19_autograd.py. La usan las
lecciones 4.1 (para comprobarla), 4.2 (bloque transformer) y 4.3
(mini-GPT).
"""

import numpy as np


def _desbroadcast(grad, forma):
    """Si una operacion hizo broadcasting, el gradiente tiene la forma
    "agrandada": hay que sumarlo de vuelta a la forma original.
    Ejemplo: b de forma (4,) sumado a X de forma (200, 4) -> el gradiente
    de b es la suma sobre las 200 filas (como np.sum(dZ, axis=0) en la 1.3)."""
    while grad.ndim > len(forma):
        grad = grad.sum(axis=0)
    for eje, n in enumerate(forma):
        if n == 1 and grad.shape[eje] != 1:
            grad = grad.sum(axis=eje, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _previos=(), nombre=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._previos = _previos
        self._atras = lambda: None
        self.nombre = nombre

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}{', ' + self.nombre if self.nombre else ''})"

    # ---------- operaciones basicas ----------
    def __add__(self, otro):
        otro = otro if isinstance(otro, Tensor) else Tensor(otro)
        salida = Tensor(self.data + otro.data, (self, otro))

        def _atras():
            self.grad += _desbroadcast(salida.grad, self.shape)
            otro.grad += _desbroadcast(salida.grad, otro.shape)
        salida._atras = _atras
        return salida

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Tensor) else Tensor(otro)
        salida = Tensor(self.data * otro.data, (self, otro))

        def _atras():
            self.grad += _desbroadcast(otro.data * salida.grad, self.shape)
            otro.grad += _desbroadcast(self.data * salida.grad, otro.shape)
        salida._atras = _atras
        return salida

    def __matmul__(self, otro):
        salida = Tensor(self.data @ otro.data, (self, otro))

        def _atras():
            # (..., n, k) @ (..., k, m): las mismas formulas de la 1.3 (X.T @ delta)
            self.grad += _desbroadcast(salida.grad @ np.swapaxes(otro.data, -1, -2), self.shape)
            otro.grad += _desbroadcast(np.swapaxes(self.data, -1, -2) @ salida.grad, otro.shape)
        salida._atras = _atras
        return salida

    def __pow__(self, k):
        salida = Tensor(self.data ** k, (self,))

        def _atras():
            self.grad += k * self.data ** (k - 1) * salida.grad
        salida._atras = _atras
        return salida

    def __neg__(self):
        return self * -1.0

    def __sub__(self, otro):
        return self + (-otro if isinstance(otro, Tensor) else Tensor(-np.asarray(otro, dtype=float)))

    def __truediv__(self, otro):
        return self * (otro ** -1.0 if isinstance(otro, Tensor) else 1.0 / otro)

    __radd__ = __add__
    __rmul__ = __mul__

    # ---------- funciones ----------
    def tanh(self):
        t = np.tanh(self.data)
        salida = Tensor(t, (self,))

        def _atras():
            self.grad += (1 - t ** 2) * salida.grad          # leccion 1.5
        salida._atras = _atras
        return salida

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        salida = Tensor(s, (self,))

        def _atras():
            self.grad += s * (1 - s) * salida.grad           # leccion 1.1
        salida._atras = _atras
        return salida

    def relu(self):
        salida = Tensor(np.maximum(0, self.data), (self,))

        def _atras():
            self.grad += (self.data > 0) * salida.grad       # leccion 1.5
        salida._atras = _atras
        return salida

    def exp(self):
        e = np.exp(self.data)
        salida = Tensor(e, (self,))

        def _atras():
            self.grad += e * salida.grad
        salida._atras = _atras
        return salida

    def log(self):
        salida = Tensor(np.log(self.data), (self,))

        def _atras():
            self.grad += salida.grad / self.data
        salida._atras = _atras
        return salida

    def sum(self, axis=None, keepdims=False):
        salida = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _atras():
            g = salida.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)      # cada sumando recibe el mismo gradiente
        salida._atras = _atras
        return salida

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=axis, keepdims=True)
        salida = Tensor(p, (self,))

        def _atras():
            g = salida.grad                                   # derivada de softmax (leccion 3.6)
            self.grad += p * (g - (p * g).sum(axis=axis, keepdims=True))
        salida._atras = _atras
        return salida

    # ---------- formas e indices ----------
    def reshape(self, *forma):
        salida = Tensor(self.data.reshape(*forma), (self,))

        def _atras():
            self.grad += salida.grad.reshape(self.shape)
        salida._atras = _atras
        return salida

    def transpose(self, *ejes):
        salida = Tensor(self.data.transpose(*ejes), (self,))

        def _atras():
            self.grad += salida.grad.transpose(np.argsort(ejes))
        salida._atras = _atras
        return salida

    def __getitem__(self, indices):
        salida = Tensor(self.data[indices], (self,))

        def _atras():
            np.add.at(self.grad, indices, salida.grad)        # filas repetidas suman (leccion 3.4)
        salida._atras = _atras
        return salida

    # ---------- el backward ----------
    def backward(self):
        """Ordena el grafo (orden topologico) y aplica la regla de la cadena
        desde este tensor hacia atras."""
        orden, visitados = [], set()

        def construir(t):
            if id(t) not in visitados:
                visitados.add(id(t))
                for p in t._previos:
                    construir(p)
                orden.append(t)
        construir(self)
        self.grad = np.ones_like(self.data)                   # dL/dL = 1
        for t in reversed(orden):
            t._atras()


def cross_entropy(logits, objetivos):
    """Promedio de -log softmax(logits)[objetivo]. Fusionada porque su
    gradiente es simplemente (probs - one_hot) / n (leccion 1.4)."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=-1, keepdims=True)
    planas = probs.reshape(-1, probs.shape[-1])
    obj = np.asarray(objetivos).reshape(-1)
    n = len(obj)
    salida = Tensor(-np.mean(np.log(planas[np.arange(n), obj] + 1e-12)), (logits,))

    def _atras():
        g = planas.copy()
        g[np.arange(n), obj] -= 1
        logits.grad += (g / n).reshape(logits.shape) * salida.grad
    salida._atras = _atras
    return salida


def parametros_en_cero(parametros):
    for p in parametros:
        p.grad = np.zeros_like(p.data)


class Adam:
    """Descenso de gradiente con tasa propia por peso (como Adagrad, leccion
    3.5) y "inercia": promedia los gradientes recientes en vez de sumarlos
    todos. Es el optimizador que se usa para entrenar transformers."""

    def __init__(self, parametros, tasa=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parametros = parametros
        self.tasa, self.beta1, self.beta2, self.eps = tasa, beta1, beta2, eps
        self.m = [np.zeros_like(p.data) for p in parametros]   # promedio del gradiente
        self.v = [np.zeros_like(p.data) for p in parametros]   # promedio del gradiente al cuadrado
        self.t = 0

    def paso(self):
        self.t += 1
        for p, m, v in zip(self.parametros, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * p.grad
            v[:] = self.beta2 * v + (1 - self.beta2) * p.grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)              # correccion: al inicio m y v valen ~0
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data -= self.tasa * m_hat / (np.sqrt(v_hat) + self.eps)
