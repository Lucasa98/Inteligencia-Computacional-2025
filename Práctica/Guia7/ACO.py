import numpy as np

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
        self.costos[i,j] = np.power(self.costos[i,j] / G[i,j], self.β)   # amplificar por beta
        self.σ[i,j] = self.rng.uniform(0, 1, size=len(i))
        # reflejar diagonal superior en la inferior
        self.costos[j,i] = self.costos[i,j]
        self.σ[j,i] = self.σ[i,j]

    def buscar(self, G: np.ndarray[float], A: int, B: int):
        """
        Ejecuta ACO para buscar un camino desde A hasta B 
        Retorna (best_path, best_length)
        - best_path: lista de aristas [(i,j), (j,k), ...]
        - best_length: suma de distancias
        """
        # inicializar feromonas con valores al azar
        # y costos para cada camino
        self.init(G)
        # ubicar hormigas en el origen
        self.hormigas = [[] for _ in range(self.N)]
        # reservar memoria para longitud de caminos
        long = np.zeros(self.N, dtype=int)

        best_long = np.inf
        best_path = None
        no_improve = 0
        it = 0
        it_max_improve = 5

        #condición de corte: 5 iteraciones sin mejora (por lo tanto siguen el mismo camino) o max_it iteraciones
        while no_improve<it_max_improve and it < self.max_it:
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

                    # anular probabilidades de nodos ya visitados
                    for i in range(G.shape[0]):
                        if i in recorridos:
                            probs[i] = 0.0

                    total = probs.sum()
                    if total == 0:
                        raise ValueError(f"No hay caminos disponibles desde {a}")

                    # seleccionar proximo nodo
                    next = self.rng.choice(np.arange(G.shape[0]), p=probs/total)   # elegir con probabilidades
                    hormiga.append((a,next))
                    recorridos.append(next)

                # calcular longitud del camino
                self.hormigas[k] = hormiga
                long[k] = sum(G[i,j] for i, j in hormiga)
            
            #hormiga elite
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

            match self.metodo:
                case 'uniforme':
                    for k in range(len(self.hormigas)):
                        for i, j in self.hormigas[k]:
                            self.σ[i,j] += self.Q
                            self.σ[j,i] = self.σ[i,j]

                case 'local':   
                    for k in range(len(self.hormigas)):
                        for i, j in self.hormigas[k]:
                            self.σ[i,j] += self.Q/long[k]
                            self.σ[j,i] = self.σ[i,j]
                
                case 'global':
                    for i, j in best_path:
                        self.σ[i,j] += self.Q/best_long
                        self.σ[j,i] = self.σ[i,j]
                
                case _:
                    raise ValueError(f"Método {self.metodo} no reconocido")


            if it%10 == 0:
                print(f"Iteración {it}: mejor longitud = {best_long:.3f}")
            
            
            it += 1

        print(f"Terminado en it {it} - mejor longitud = {best_long:.3f} - camino = {best_path}")

        self.best_long = best_long
        self.best_path = best_path
        return self.best_long, self.best_path
