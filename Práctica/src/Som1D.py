import numpy as np

class Som1D:
    def __init__(self, N: int, tasa_aprendizaje: float):
        self.η = tasa_aprendizaje
        self.N = N
        self.rng = np.random.default_rng()
        self.history: list[np.ndarray[float]] = []

    def entrenar(self, x: np.ndarray[float], max_epocas: int = 10):
        # inicializar pesos
        W = self.rng.uniform(-0.5, 0.5, (self.N, x.shape[1]))
        entorno = 1

        for _ in range(max_epocas):
            for i in range(x.shape[0]):
                # encontrar ganadora G = argmin(dist(W,x[i]))
                Gi = np.argmin(np.sum(np.power(W - x[i],2), 1))
                # definir vecinas
                ini, fin = max(0,Gi-entorno), min(self.N, Gi+entorno+1)
                # ajustar ganadora y vecinas
                W[ini:fin] = W[ini:fin] + self.η * (x[i] - W[ini:fin])
                self.history.append(W.copy())

        self.W = W