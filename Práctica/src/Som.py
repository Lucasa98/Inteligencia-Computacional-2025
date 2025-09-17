import numpy as np

class Som:
    def __init__(self, N: int, vecindades: list[tuple[float]], tasa_aprendizaje: list[tuple[float]], epocas: list[int]):
        """
        Args:
            N (int): dimension del plano de neuronas
            tasa_aprendizaje (list[list[float]]): lista de tasas de aprendizaje inicial y final por cada etapa
            epocas (list[int]): epocas por cada etapa
        """
        self.η = tasa_aprendizaje
        self.N = N
        self.epocas = epocas
        self.vecindades = vecindades
        self.rng = np.random.default_rng()
        self.history: list[np.ndarray[float]] = []

    def entrenar(self, x: np.ndarray[float]) -> list[np.ndarray[float]]:
        W_etapas = []
        # inicializar pesos (N x N x entradas)
        W = self.rng.uniform(-0.5, 0.5, (self.N, self.N, x.shape[1]))

        # el entrenamiento se divide en etapas, cada una con una cantidad de epocas y una tasa de aprendizaje inicial y final
        for j in range(len(self.epocas)):
            ini_entorno = self.vecindades[j][0]
            d_ent = (self.vecindades[j][1]-ini_entorno)/self.epocas[j]    # decaimiento de entorno
            apr = self.η[j][0]                          # tasa de aprendizaje inicial
            d_apr = (apr - self.η[j][1])/self.epocas[j] # decaimiento de la tasa de aprendizaje

            for e in range(self.epocas[j]):
                entorno = round(ini_entorno + e*d_ent)
                for i in range(x.shape[0]):
                    diff = x[i] - W
                    # encontrar ganadora G = argmin(dist(W,x[i]))
                    dist = np.sum(np.power(diff, 2), 2)
                    Gix, Giy = np.unravel_index(np.argmin(dist), dist.shape)
                    # definir vecinas
                    inix, finx, iniy, finy = max(0,Gix-entorno), min(self.N, Gix+entorno+1), max(0,Giy-entorno), min(self.N, Giy+entorno+1)
                    # ajustar ganadora y vecinas
                    W[inix:finx, iniy:finy] = W[inix:finx, iniy:finy] + apr * diff[inix:finx, iniy:finy]
                self.history.append(W.copy())
                apr -= d_apr

            W_etapas.append(W.copy())

        self.W = W
        return W_etapas

    def etiquetar(self, x: np.ndarray[float], y: np.ndarray[any]):
        etiquetas = np.empty(shape=(self.N, self.N), dtype=type(y.dtype))
        for i in range(x.shape[0]):
            # encontrar ganadora G = argmin(dist(W,x[i]))
            dist = np.sum(np.power(x[i] - self.W, 2), 2)
            # TODO: hacer que se etiquete por la mas frecuente, no la ultima
            etiquetas[np.unravel_index(np.argmin(dist), dist.shape)] = y[i]

        self.tags = etiquetas

    def predict(self, x: np.ndarray[float]) -> np.ndarray[float]:
        y = np.empty(shape=x.shape, dtype=float)
        for i in range(x.shape[0]):
            # encontrar ganadora G = argmin(dist(W,x[i]))
            dist = np.sum(np.power(x[i] - self.W, 2), 2)
            y[i] = self.W[np.unravel_index(np.argmin(dist), dist.shape)]

        return y
