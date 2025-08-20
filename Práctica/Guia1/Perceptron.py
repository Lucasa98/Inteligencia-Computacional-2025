import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    trainingGif()
        Genera un gif con la evolucion de las entradas en el entrenamiento (solo 2 entradas)
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
        self.error_history = []

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
        w = self.w + self.rate * error * x
        self.c_ajustes += 1

        # solo agregar al historial si es distinto del actual
        if error != 0:
            self.addW(w)

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

    def entrenar(self, x, yd, gifPath = None, label = '', targetError = -1) -> float:
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
            self.error_history.append(error)
            if error <= targetError:
                break

        self.errorEvol(label=label)

        if gifPath != None:
            self.trainingGif(x,yd,gifPath,label)

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

    def trainingGif(self, x, yd, gifPath: str, gifLabel: str = ''):
        fig, ax = plt.subplots()
        ax.set(
            xlim=[-2,2],
            ylim=[-2,2],
            xlabel='x1',
            ylabel='x2',
            title=f"Evolucion del perceptron {gifLabel}"
        )

        # puntos
        ax.scatter(x[yd == -1, 0], x[yd == -1, 1], color='red')
        ax.scatter(x[yd == 1, 0], x[yd == 1, 1], color='green')

        # rectas
        x_vals = np.linspace(-2, 2, 100)
        w1, w2, w0 = self.W[0][1]
        # para evitar division por cero
        if w2 == 0:
            y_vals = np.full_like(x_vals, np.nan)
        else:
            # y_vals = (-w1/w2) * x_vals + (w0/w2)
            y_vals = (-w1/w2) * x_vals + (w0/w2)
        recta, = ax.plot(x_vals, y_vals)
        def update(frame):
            if (frame >= len(self.W)):
                return update(len(self.W) - 1)
            w1, w2, w0 = self.W[frame][1]
            if w2 == 0:
                y_vals = np.full_like(x_vals, np.nan)
            else:
                y_vals = (-w1/w2) * x_vals + (w0/w2)
            recta.set_ydata(y_vals)
            return recta,

        gif = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=len(self.W) + 3, # espera al final
            interval=20,
            repeat=False,
            blit=True,  # pequena optimizacion
        )

        gifWriter = animation.PillowWriter(fps=2)
        gif.save(gifPath, writer=gifWriter)

    def errorEvol(self, label=''):
        fig, ax = plt.subplots()
        ax.set(
            xlabel='epoca',
            ylabel='% error',
            title=f"Evolucion del error {label}"
        )

        errores = self.error_history if len(self.error_history) > 1 else [self.error_history[0], self.error_history[0]] # para mostrar algo si se llego en una sola epoca
        ax.plot(errores)

        ax.grid()
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
