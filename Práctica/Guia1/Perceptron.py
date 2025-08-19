import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Perceptron simple con un numero fijo de entradas.

    Atributos
    ------
    W : dict[int, np.ndarray]
        historial de pesos, asociados con la iteracion te entrenamiento en la que se generaron. El ultimo es w
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
        self.max_epocas = maxEpocas
        self.bias = bias
        self.rng = np.random.default_rng()
        self.phi = funcionActivacion
        self.N = N + 1 # entradas + bias
        self.c_ajustes = 0
        self.initW()

    @property
    def w(self) -> np.ndarray[float]:
        """Vector de pesos actual

        Raises:
            RuntimeError: si los pesos no estan inicializados

        Returns:
            NDArray[float]: lista de pesos actuales
        """
        if len(self.W) == 0:
            raise RuntimeError('Pesos no inicializados')

        return self.W[-1][1]

    def addW(self, w: np.ndarray):
        """Actualiza la lista de pesos agregando N pesos nuevos, que pasan a ser los actuales

        Args:
            w (list[float] | np.ndarray): pesos a agregar

        Raises:
            TypeError: el numero de pesos no coincide con el numero de entradas (N)
        """
        if w.shape[0] != self.N:
            raise TypeError(f"Se esperaban {self.N} pesos. w.shape[0]={w.shape[0]}")

        self.c_ajustes += 1

        # solo agregar al historial si es distinto del actual
        if not np.array_equal(self.w, w):
            self.W.append((self.c_ajustes, w))

    def initW(self):
        """Inicializa los pesos de forma aleatoria en [-0.5,0.5)"""
        w_init = self.rng.uniform(-0.5, 0.5, self.N)
        if self.bias != None:   # Si se especifica un bias, se empieza con ese valor
            self.w[-1] = self.bias
        self.W = [(0, w_init)]

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

    def entrenar(self, x, yd, targetError = -1) -> float:
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
        for i in range(self.max_epocas):
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

    def graphTraining(self, limit = -1):
        if self.N != 3:
            raise TypeError('No se puede graficar para mas de 2 entradas (+bias)')

        x_vals = np.linspace(-2, 2, 100)

        plt.figure(figsize=(10,10))

        W = self.W if limit == -1 else [w for i in ]

        for it, w in self.W:
            w1 = w[0]
            w2 = w[1]
            w0 = w[2]
            if w1 == 0:
                continue

            # w1x1 + w2x2 - w0 = 0 => x2 = -w1x1/w2 + w0/w2
            y_vals = (-w1/w2) * x_vals + (w0 / w2)
            plt.plot(x_vals, y_vals, label=f"Ajuste {int(it)}")

        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Evolucion del perceptron")
        plt.legend()
        plt.grid()
        plt.show()

    def graphWeightEvol(self):
        plt.figure(figsize=(15,5))

        it_values = [it for it,_ in self.W]

        w0 = [w[-1] for _,w in self.W]
        plt.plot(it_values, w0, label=f"w0")
        for i in range(self.N-1):
            wi = [w[i] for _,w in self.W]
            
            plt.plot(it_values, wi, label=f"w{i+1}")

        plt.xlabel('iteracion')
        plt.ylabel('peso')
        plt.legend()
        plt.grid()
        plt.show()
