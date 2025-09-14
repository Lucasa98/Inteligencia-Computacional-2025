import numpy as np

class Kmeans:
    def __init__(self, N: int):
        self.N = N
        self.rng = np.random.default_rng()

    def entrenar(self, x: np.ndarray[float]):
        patrones = x.shape[0]
        # inicializar centroides en un patron aleatorio
        self.centroides = x[self.rng.integers(0, patrones, self.N), :].copy()
        # crear vector de asignacion de grupo
        grupos = np.zeros(patrones, dtype=int)
        tmpGrupos = np.zeros(patrones, dtype=int)

        # asignamos memoria para los vectores de "distancias" entre cada centroide y los patrones
        dist = np.zeros((patrones, self.N), dtype=float)
        while True:
            # calcular "distancias" (no calculamos la raiz cuadrada)
            for i in range(self.N):
                dist[:,i] = np.sum(np.power(x - self.centroides[i], 2), axis=1)

            # calcular el grupo para cada patron
            for i in range(patrones):
                tmpGrupos[i] = np.argmin(dist[i,:])

            # si no hubo reasignaciones, terminar
            if (tmpGrupos == grupos).all():
                break;

            # recalcular cada centroide
            grupos = tmpGrupos.copy()
            for i in range(self.N):
                self.centroides[i] = x[(grupos == i)].mean(axis=0)
