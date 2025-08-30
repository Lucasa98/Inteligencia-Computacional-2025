import numpy as np

class PerceptronMulticapa:
    def __init__(self, cant_entradas, capas: list[int], max_epocas: int, tasa_aprendizaje):
        """
        Args:
            capas (list[int]): arreglo con la cantidad de neuronas de cada 
        """
        self.cant_entradas = cant_entradas
        self.max_epocas = max_epocas
        self.capas = capas
        self.η = tasa_aprendizaje
        self.W : list[np.ndarray] = []
        self.error_history = []
        self.quaderror_history = []

        # iniciar generador de numeros
        self.rng = np.random.default_rng()

        # init pesos
        for capa in range(len(self.capas)):
            # las entradas de la primer capa son las entradas de la red
            cant_pesos = cant_entradas if capa == 0 else self.capas[capa-1]
            Wcapa = self.rng.uniform(-0.5, 0.5, (self.capas[capa], cant_pesos + 1)) # le sumamos un peso (del bias)
            self.W.append(Wcapa)

    def entrenar(self, x: np.ndarray[np.ndarray[float]], yd: np.ndarray[float], targetError: float = -1) -> float:
        patrones = x.shape[0]

        if x.shape[1] != self.cant_entradas:
            raise TypeError(f"Se esperaba {self.cant_entradas}. x contiene {x.shape[1]} entradas por patron.")

        # agregar entrada -1 del bias
        x = np.hstack([x, -1 * np.ones((patrones, 1))])

        error = 0.0
        for i in range(self.max_epocas):
            # iterar por patrones
            quad_error = 0
            for n in range(patrones):
                # guardamos las entradas de cada capa (la salida de la anterior y -1 del bias, que se lo agregue antes)
                entradas: list[np.ndarray[float]] = [x[n]]
                # guardamos las salidas de cada capa (estamos repitiendo datos que estan en `entradas` (sin el -1), pero es para claridad)
                y = []
                # PROPAGACION HACIA ADELANTE (calculo de salidas)
                for j in range(len(self.capas)):
                    # salida lineal
                    v = np.dot(self.W[j],entradas[j])
                    entradas.append(np.append(self.φ(v),-1))
                    y.append(self.φ(v))

                # Calculo del error
                e: np.ndarray[float] = np.subtract(yd[n],y[-1]) # vector de errores en salida
                ξ = 0.5 * np.sum(np.power(e,2))             # error cuadratico total
                quad_error += ξ

                # RETROPROPAGACION (calculo de deltas)
                deltas = [None] * len(self.capas)     # lista de deltas de cada capa
                deltas[-1] = e * self.dφ(y[-1])     # delta de ultima capa
                for j in range(len(self.capas)-2, -1, -1):
                    # el vector de deltas de la capa j es el W[j+1]^T * deltas[j+1] * φ'(y[j])
                    retro = np.dot(self.W[j+1].T, deltas[j+1])
                    retro = retro[:-1]      # descartamos el ultimo (del bias)
                    deltas[j] = retro * self.dφ(y[j])

                # calcular y aplicar ajustes
                for j in range(len(self.capas)):
                    dW = self.η * np.outer(deltas[j], entradas[j])  # para algo servia el vector entradas
                    self.W[j] += dW

            # verificar
            error = self.errorRate(x, yd)
            self.error_history.append(error)
            self.quaderror_history.append(quad_error/patrones)
            if error <= targetError:
                break

        return error

    def test(self, x: np.ndarray[np.ndarray[float]], yd: np.ndarray[float]) -> float:
        patrones = x.shape[0]

        if x.shape[1] != self.cant_entradas:
            raise TypeError(f"Se esperaba {self.cant_entradas}. x contiene {x.shape[1]} entradas por patron.")

        # agregar entrada -1 del bias
        x = np.hstack([x, -1 * np.ones((patrones, 1))])

        return self.errorRate(x, yd)

    def calcular(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """Calcular la salida binarizada (-1/+1) de la red dada una entrada
        Args:
            x (np.ndarray[float]): entradas
        Returns:
            np.ndarray[float]: salidas
        """
        # entradas para cada capa ()
        entradas : list[np.ndarray[float]] = [np.append(x,-1)] if x.shape[0] == self.cant_entradas-1 else [x]

        # PROPAGACION HACIA ADELANTE
        y = entradas[0]
        for capa in range(len(self.capas)):
            # salida lineal
            v = np.dot(self.W[capa], entradas[capa])
            # salida no-lineal
            y = self.φ(v)
            # guardar salida con -1 (para el bias de la siguiente capa)
            entradas.append(np.append(y, -1))

        if y.shape[0] == 1:
            return np.ones(y.shape, dtype=float) if y >= 0 else -1 * np.ones(y.shape, dtype=float)
        y_wta = -1 * np.ones(y.shape, dtype=float)
        y_wta[np.argmax(y)] = 1
        return y_wta

    def φ(self, v):
        return 2.0 / (1.0 + np.exp(-v)) - 1.0

    def dφ(self, y):
        """Derivada de la funcion de activacion"""
        return 0.5 * (1 + y) * (1 - y)

    def delta(self, error, y):
        return 0.5 * error * (1 + y) * (1 - y)

    def errorRate(self, x: np.ndarray[np.ndarray[float]], yd: np.ndarray[float]) -> float:
        """Calcula la tasa de error del perceptron frente a unas entradas y salidas. Las entradas deben incluir el -1 de bias al final"""
        patrones = x.shape[0]
        fallos = 0
        for i in range(patrones):
            y = self.calcular(x[i])
            if (y!=yd[i]).any():    # mmm esto solo para el caso de xor
                fallos += 1

        return fallos/patrones

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute output of the NN for inputs x.
        x shape: (N, cant_entradas)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        patrones = x.shape[0]

        # bias
        x = np.hstack([x, -1 * np.ones((patrones, 1))])

        y_final = []
        for n in range(patrones):
            entradas = x[n]
            for j in range(len(self.capas)):
                v = np.dot(self.W[j], entradas)
                salida = self.φ(v)
                # bias
                entradas = np.append(salida, -1)
            y_final.append(salida)

        return np.array(y_final).squeeze()
