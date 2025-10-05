import numpy as np

class ACO:
    def __init__(self, cant_hormigas: int, tasa_evaporacion: float, feromonas_depositadas: float, metodo: str = 'global', alpha: float = 1.0, beta: float = 1.0):
        self.N = cant_hormigas
        self.eta = 1.0-tasa_evaporacion # (1 - η)
        self.Q = feromonas_depositadas
        self.metodo = metodo
        self.α = alpha                  # amplificacion de las hormonas de un camino para el deseo
        self.β = beta                   # amplificacion del costo de un camino para el deseo
        self.rng = np.random.default_rng()

    def init(self,G: np.ndarray[float]):
        N = G.shape[0]
        # matriz igual a G pero con 0s
        self.σ = np.ones_like(G)
        self.costos = np.ones_like(G)
        # indices del triangulo superior de una matriz NxN, con offset 1 (no queremos la diagonal)
        i, j = np.triu_indices(N,k=1)
        # inicializar diagonal superior
        self.costos[i,j] = np.power(self.costos[i,j] / G[i,j], self.β)   # amplificar por beta
        self.σ[i,j] = self.rng.uniform(0, 1, size=len(i))
        # reflejar diagonal superior en la inferior
        self.costos[j,i] = self.costos[i,j]
        self.σ[j,i] = self.σ[i,j]

    def buscar(self, G: np.ndarray[float], A: int, B: int):
        # inicializar feromonas con valores al azar
        # y costos para cada camino
        self.init(G)
        # ubicar hormigas en el origen
        self.hormigas = [[] for _ in range(self.N)]
        # reservar memoria para longitud de caminos
        long = np.zeros(self.N, dtype=int)

        # hasta que todas las hormigas sigan el mismo camino
        iteracion = 0
        while True:
            # calcular deseo de cada camino
            deseo = np.power(self.σ,self.α)*self.costos
            # Por cada hormiga
            for k in range(len(self.hormigas)):
                # vaciar el camino
                hormiga = []
                # vaciar recorridos
                recorridos = [A]
                # repetir hasta que la hormiga llegue a destino
                while recorridos[-1] != B:
                    # nodo actual
                    a = recorridos[-1]

                    # probabilidad de cada nodo
                    probs = deseo[a,:].copy()
                    for i in range(G.shape[0]):
                        if i in recorridos:
                            probs[i] = 0.0
                    total = probs.sum()
                    if total == 0:
                        raise ValueError(f"No hay caminos disponibles desde {a}")

                    # seleccionar proximo nodo
                    siguiente = self.rng.choice(np.arange(G.shape[0]), p=probs/total)   # elegir con probabilidades
                    hormiga.append((a,siguiente))
                    recorridos.append(siguiente)

                # calcular longitud del camino
                self.hormigas[k] = hormiga
                long[k] = sum(G[i,j] for i, j in hormiga)

            # condicion de corte (todas las hormigas siguen el mismo camino)
            k = 1
            while k < len(self.hormigas) and self.hormigas[k] == self.hormigas[k-1]:
                k += 1
            if k == len(self.hormigas):
                break;
            if iteracion % 100 == 0:
                print(f"iteracion: {iteracion}")

            # evaporar
            self.σ *= self.eta

            if self.metodo == 'uniforme':
                for hormiga in self.hormigas:
                    for i,j in hormiga:
                        self.σ[i,j] += self.Q
                        self.σ[j,i] = self.σ[i,j]
            # TODO: implementar los otros metodos. Capaz lo mejor seria calcular los deltas por adelantado
            iteracion += 1

        print(f"Terminado en iteracion {iteracion}")
        return self.hormigas[0]
