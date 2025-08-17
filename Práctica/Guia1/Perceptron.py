import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Perceptron simple con un numero fijo de entradas.

    Atributos
    ------
    w : array[float]
        vector de pesos para las entradas + umbral.

    Metodos
    ------
    entrenar(x, yd, targetError = 0)
        Entrena el perceptron con unas entradas y salidas esperadas

    test(x, yd)
        Testea el perceptron y devuelve la tasa de error

    graphTraining()
        Imprime un grafico con la evolucion de los pesos a lo largo del entrenamiento
    """

    def __init__(self, N: int, rate: float, maxEpocas: int, funcionActivacion: callable, bias: float = None):
        """
        Args:
            N (int): numero de entradas
            rate (float): tasa de aprendizaje
            maxEpocas (int): numero maximo de epocas
            funcionActivacion (callable): funcion de activacion a utilizar
            bias (float, optional): umbral de activacion inicial
        """
        self.rate = rate
        self.maxEpocas = maxEpocas
        self.bias = bias
        self.rng = np.random.default_rng()
        self.phi = funcionActivacion
        self.N = N + 1 # entradas + bias
        self.initW()

    @property
    def w(self) -> np.ndarray[float]:
        """Vector de pesos actual

        Raises:
            RuntimeError: _description_

        Returns:
            NDArray[float]: _description_
        """
        if self.W.shape[0] == 0:
            raise RuntimeError('Pesos no inicializados')

        return self.W[-1]

    def addW(self, w):
        """Actualiza la lista de pesos agregando N pesos nuevos, que pasan a ser los actuales

        Args:
            w (list[float] | np.ndarray): pesos a agregar

        Raises:
            TypeError: el numero de pesos no coincide con el numero de entradas (N)
        """
        if w.shape[0] != self.N:
            raise TypeError(f"Se esperaban {self.N} pesos. w.shape[0]={w.shape[0]}")

        # solo agregar al historial si es distinto del actual
        if not np.array_equal(self.w, w):
            self.W = np.vstack([self.W, w])

    def initW(self):
        """Inicializa los pesos de forma aleatoria en [-0.5,0.5)"""
        self.W = np.empty((0,self.N), dtype=float)
        w_init = self.rng.uniform(-0.5, 0.5, self.N)
        if self.bias != None:   # Si se especifica un bias, se empiza con ese valor
            self.w[-1] = self.bias
        self.W = np.vstack([self.W, w_init])

    def suma(self, x) -> float:
        """suma ponderada

        Args:
            x (list[float]): entradas

        Raises:
            TypeError: error de tamanio de pesos y entradas

        Returns:
            float: suma ponderada
        """
        if x.shape[0] != self.N:
            raise TypeError(f"Se esperaban {self.N} entradas. x.shape[0]={x.shape[0]}")

        return np.dot(self.w,x)

    def calcular(self, x) -> float:
        """Salida del perceptron segun su funcion de activacion

        Args:
            x (list[float]): entradas

        Returns:
            float: salida
        """
        if self.N != x.shape[0]:
            raise TypeError(f"Se esperaban {self.N} entradas. x.shape[0]={x.shape[0]}")

        return self.phi(self.suma(x))

    def ajustarPesos(self, x, error):
        """ajusta los pesos segun las entradas y el error de la prediccion

        Args:
            x (list[float]): entradas
            error (float): error de la prediccion
        """
        self.addW(self.w + self.rate * error * x)

    def errorRate(self, x, yd) -> float:
        """Tasa de error

        Args:
            x (list[list[float]]): entradas
            yd (list[float]): salidas esperadas

        Returns:
            float: tasa de error
        """
        casos = x.shape[0]
        fallos = 0
        for n in range(casos):
            y = self.calcular(x[n])
            if y != yd[n]:
                fallos += 1

        return fallos/casos

    def entrenar(self, x, yd, targetError = 0) -> float:
        """Entrenar perceptron con datos de entradas y salidas esperadas

        Args:
            x (list[list[float]]): entradas
            yd (list[float]): salidas esperadas
            targetError (float, optional): criterio de corte, si no se especifica se realiza el numero maximo de epocas. Por defecto 0.

        Returns:
            float: tasa de error final segun datos de entrenamiento
        """
        # Inicializar pesos
        casos = x.shape[0]

        # agregar entrada del bias
        x = np.hstack([x, -1 * np.ones((casos, 1))])

        if self.N != x.shape[1]:
            raise TypeError(f"Se esperaban {self.N} entradas. x.shape[1]={x.shape[1]}")

        error = 0.0
        for i in range(self.maxEpocas):
            # ajustar con datos de training
            for n in range(casos):
                # calcular salida
                y = self.calcular(x[n])
                # ajustar
                self.ajustarPesos(x[n],yd[n]-y)

            # verificar
            error = self.errorRate(x,yd)
            if error <= targetError:
                break

        return error

    def test(self, x, yd) -> float:
        """Testear perceptron con datos y salidas esperadas

        Args:
            x (list[list[float]]): entradas
            yd (list[float]): salidas esperadas

        Returns:
            float: tasa de error
        """
        if self.N-1 != x.shape[1]:
            raise TypeError(f"Se esperaban {self.N} entradas. x.shape[1]={x.shape[1]}")

        # agregar entrada del bias
        x = np.hstack([x, -1 * np.ones((x.shape[0], 1))])

        return self.errorRate(x,yd)

    def graphTraining(self):
        if self.N != 3:
            raise TypeError('No se puede graficar para mas de 2 entradas (+bias)')

        x_vals = np.linspace(-1, 1, 100)

        plt.figure(figsize=(15,10))

        for i, (w0, w1, w2) in enumerate(self.W):
            if w1 == 0:
                continue

            y_vals = -(w0/w1) * x_vals - (w2 / w1)
            plt.plot(x_vals, y_vals, label=f"Ajuste {i}")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Evolucion del perceptron")
        plt.legend()
        plt.grid()
        plt.show()
