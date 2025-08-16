import numpy as np

class Perceptron:
    def __init__(self, rate, maxEpocas, funcionActivacion, bias = None):
        """Perceptron simple

        Args:
            rate (float): tasa de aprendizaje
            maxEpocas (int): numero maximo de epocas
            bias (float): umbral de activacion inicial
            funcionActivacion (callable): funcion de activacion a utilizar
        """
        self.rate = rate
        self.maxEpocas = maxEpocas
        self.bias = bias
        self.rng = np.random.default_rng()
        self.phi = funcionActivacion
        self.w = []

    def initW(self, N):
        """Inicializa los pesos de forma aleatoria en [-0.5,0.5)

        Args:
            N (int): numero de pesos
        """
        self.N = N + 1
        self.w = self.rng.uniform(-0.5, 0.5, N)
        if self.bias != None:
            self.w[-1] = self.bias

    def suma(self, x):
        """suma ponderada

        Args:
            x (array<float>): entradas

        Raises:
            TypeError: error de tamanio de pesos y entradas

        Returns:
            float: suma ponderada
        """
        if len(x) != len(self.w):
            raise TypeError(f"El numero de entradas no coincide con las del perceptron. len(x)={len(x)}, len(self.w)={len(self.w)}")

        return np.dot(self.w,x)

    def calcular(self, x):
        """Salida del perceptron segun su funcion de activacion

        Args:
            x (array<float>): entradas

        Returns:
            float: salida
        """
        return self.phi(self.suma(x))

    def ajustarPesos(self, x, error):
        """ajusta los pesos segun las entradas y el error de la prediccion

        Args:
            x (arrat<float>): entradas
            error (float): error de la prediccion
        """
        self.w = self.w + self.rate * error * x

    def errorRate(self, x, yd):
        """Tasa de error

        Args:
            x (array<array<float>>): entradas
            yd (arrat<float>): salidas esperadas

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

    def entrenar(self, x, yd, targetError = 0):
        """Entrenas perceptron con datos de entradas y salidas esperadas

        Args:
            x (array<array<float>>): entradas
            yd (array<float>): salidas esperadas
            targetError (float, optional): especificar si se quiere especificar una tasa de error, si no realiza el maximo de epocas. Por defecto 0.

        Returns:
            float: tasa de error final segun datos de entrenamiento
        """
        # agregar entrada del bias
        x = np.hstack([x, -1 * np.ones((x.shape[0], 1))])

        casos = x.shape[0]
        self.initW(x.shape[1])

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

    def test(self, x, yd):
        """Testear perceptron con datos y salidas esperadass

        Args:
            x (array<array<float>>): entradas
            yd (arrat<float>): salidas esperadas

        Returns:
            float: tasa de error
        """
        # agregar entrada del bias
        x = np.hstack([x, -1 * np.ones((x.shape[0], 1))])

        return self.errorRate(x,yd)