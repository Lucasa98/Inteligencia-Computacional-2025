import numpy as np
import time

class ACO:
    def __init__(self, cant_hormigas: int, tasa_evaporacion: float, feromonas_depositadas: float, metodo: str = 'global', alpha: float = 1.0, beta: float = 1.0, max_it: int = 400):
        self.N = cant_hormigas
        self.eta = 1.0-tasa_evaporacion # (1 - η)
        self.Q = feromonas_depositadas
        self.metodo = str(metodo).lower().strip()
        self.α = alpha                  # amplificacion de las hormonas de un camino para el deseo
        self.β = beta                   # amplificacion del costo de un camino para el deseo
        self.max_it = max_it
        self.rng = np.random.default_rng()

    def init(self,G: np.ndarray[float]):
        N = G.shape[0]
        # matriz igual a G pero con 0s
        self.σ = np.ones_like(G)
        self.costos = np.ones_like(G)
        # indices del triangulo superior de una matriz NxN, con offset 1 (no queremos la diagonal)
        i, j = np.triu_indices(N,k=1)
        # inicializar diagonal superior
        self.costos[i, j] = np.power(1.0 / G[i, j], self.β)  # amplificar por beta. G[i,j] no va a ser nunca 0
        self.σ[i,j] = self.rng.uniform(0.1, 1, size=len(i))
        # reflejar diagonal superior en la inferior
        self.costos[j,i] = self.costos[i,j]
        self.σ[j,i] = self.σ[i,j]

    def buscar(self, G: np.ndarray[float], start: int = 0):
        """
        Ejecuta ACO para resolver el TSP:
        - start: ciudad de inicio
        Retorna (total_time, best_length, best_tour, iterations)
        """

        # inicializar feromonas con valores al azar
        # y no-costos para cada camino
        self.init(G)
        n_ciudades = G.shape[0]

        # ubicar hormigas en el origen
        self.hormigas = [[] for _ in range(self.N)]
        # reservar memoria para longitud de caminos
        long = np.zeros(self.N, dtype=float)

        best_long = np.inf
        best_path = None
        no_improve = 0
        it = 0
        it_max_no_improve = 5 
        start_time = time.time()

        # condición de corte: 5 iteraciones sin mejora
        while no_improve < it_max_no_improve and it < self.max_it:
            # calcular deseo de cada camino
            deseo = np.power(self.σ, self.α) * self.costos

            for k in range(len(self.hormigas)):
                # vaciar el camino
                hormiga = []
                pendientes = set(range(n_ciudades))
                pendientes.remove(start)
                current = start

                # repetir hasta que la hormiga haya visitado todo
                while pendientes:
                    nodos_validos = np.array(list(pendientes))
                    probs = deseo[current, nodos_validos]
                    total = probs.sum()

                    if total == 0:
                        raise ValueError(f"No hay caminos disponibles desde {current}")

                    # seleccionar siguiente ciudad
                    next_node = int(self.rng.choice(nodos_validos, p=probs/total))

                    # actualizar ya visitados
                    hormiga.append((current, next_node))
                    pendientes.remove(next_node)
                    current = next_node

                # regresar a la ciudad inicial
                hormiga.append((current, start))

                # guardamos el camino de la hormiga y su longitud
                self.hormigas[k] = hormiga
                long[k] = sum(G[i, j] for i, j in hormiga)

            # hormiga elite
            idx_best = np.argmin(long)
            best_long_actual = long[idx_best]
            best_path_actual = self.hormigas[idx_best]

            if best_long_actual < best_long:
                best_long = best_long_actual
                best_path = best_path_actual
                no_improve = 0
            else:
                no_improve += 1

            # evaporar feromonas
            self.σ *= self.eta
            self.σ = np.maximum(self.σ, 1e-10)  # evitar que feromonas lleguen a 0

            match self.metodo:
                case 'uniforme':
                    for k in range(len(self.hormigas)):
                        for i, j in self.hormigas[k]:
                            self.σ[i,j] += self.Q
                            self.σ[j,i] = self.σ[i,j]
                case 'local':
                    for k in range(len(self.hormigas)):
                        for i, j in self.hormigas[k]:
                            self.σ[i,j] += self.Q / G[i,j]
                            self.σ[j,i] = self.σ[i,j]
                case 'global':
                    for k in range(len(self.hormigas)):
                        for i, j in self.hormigas[k]:
                            self.σ[i,j] += self.Q / long[k]
                            self.σ[j,i] = self.σ[i,j]
                case _:
                    raise ValueError(f"Método {self.metodo} no reconocido")

            it += 1

        end_time = time.time()
        total_time = end_time - start_time

        # pasamos a lista de nodos
        nodos = [start]
        for i, j in best_path:
            nodos.append(j)

        return total_time, best_long, nodos, it
