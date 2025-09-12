import numpy as np

class Som:
    def __init__(self, N: int, tasa_aprendizaje: float):
        self.η = tasa_aprendizaje
        self.N = N
        self.rng = np.random.default_rng()
        self.history: list[np.ndarray[float]] = []

    def entrenar(self, x: np.ndarray[float], max_epocas: int = 10):
        # inicializar pesos (N x N x entradas)
        W = self.rng.uniform(-0.5, 0.5, (self.N, self.N, x.shape[1]))
        entorno = 1

        for _ in range(max_epocas):
            for i in range(x.shape[0]):
                diff = x[i] - W
                # encontrar ganadora G = argmin(dist(W,x[i]))
                [Gix, Giy] = np.argmin(np.sum(np.power(diff,2), 1), 0)
                # definir vecinas
                inix, finx, iniy, finy = max(0,Gix-entorno), min(self.N, Gix+entorno+1), max(0,Giy-entorno), min(self.N, Giy+entorno+1)
                # ajustar ganadora y vecinas
                W[inix:finx, iniy:finy] = W[inix:finx, iniy:finy] + self.η * diff[inix:finx, iniy:finy]
                self.history.append(W.copy())

        self.W = W
