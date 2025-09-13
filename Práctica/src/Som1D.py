import numpy as np

class Som1D:
    def __init__(self, N: int, vecindades: list[tuple[float]], tasa_aprendizaje: list[tuple[float]], epocas: list[int]):
        self.η = tasa_aprendizaje
        self.N = N
        self.epocas = epocas
        self.vecindades = vecindades
        self.rng = np.random.default_rng()
        self.history: list[np.ndarray[float]] = []

    def entrenar(self, x: np.ndarray[float]):
        W_etapas = []
        # inicializar pesos (N x entradas)
        W = self.rng.uniform(-0.5, 0.5, (self.N, x.shape[1]))

        # entrenamiento dividido en etapas
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
                    Gi = np.argmin(np.sum(np.power(diff,2), 1))
                    # definir vecinas
                    ini, fin = max(0,Gi-entorno), min(self.N, Gi+entorno+1)
                    # ajustar ganadora y vecinas
                    W[ini:fin] = W[ini:fin] + apr * diff[ini:fin]

                # guardar historial
                self.history.append(W.copy())
                apr -= d_apr

            W_etapas.append(W.copy())

        self.W = W
        return W_etapas