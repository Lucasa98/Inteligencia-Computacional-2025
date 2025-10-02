from tqdm import tqdm

import numpy as np
from typing import Callable

class Evol:
    def __init__(self,
                 init_fun: Callable[[int,int,np.random.Generator], np.ndarray[np.uint8]],
                 fit_fun: Callable[[np.ndarray[np.uint8]], np.ndarray[float]],
                 max_gen: int = 20,
                 poblacion: int = 100,
                 progenitores: int = 10,
                 long_cromosoma: int = 8,
                 mutacion: float = 0.01,
                 tolerancia: int = 5):
        """
        Args:
            init_fun: funcion para inicializar la poblacion
            fit_fun: funcion para calcular fitness

            max_gen (optional): numero maximo de generaciones. Defaults to 20.
            poblacion (optional): poblacion por generacion. Defaults to 100.
            progenitores (optional): cantidad de progenitores elegidos en cada generacion. Defaults to 10.
            precision (optional): numero de bits del cromosoma. Defaults to 8.
            mutacion (optional): probabilidad de mutacion. Defaults to 0.01.
        """
        self.init_fun = init_fun
        self.calcular_fitness = fit_fun

        self.max_gen = max_gen
        self.pob = poblacion
        self.prog = progenitores
        self.mutacion = mutacion
        self.tol = tolerancia
        self.n_bits = long_cromosoma
        self.rng = np.random.default_rng()
        self.fit_history = np.empty((0,2), dtype=float)

    def evolve(self) -> np.ndarray[float]:
        # 1) inicializar la poblacion al azar
        poblacion = self.init_fun(self.pob, self.n_bits, self.rng)
        ventana = self.pob//self.prog

        # 2) calcular fitness
        fit = self.calcular_fitness(poblacion)
        sorted = np.argsort(fit)    # indices que ordenan de menor a mayor

        c_tol: int = 0   # contador de generaciones sin mejora
        fit_elite = fit[sorted[-1]]
        self.fit_history = np.vstack([self.fit_history, [0,fit_elite]])
        for g in tqdm(range(self.max_gen)):
            # 1) elegir progenitores: un elite y el resto por ventana
            progenitores = np.empty((self.prog,self.n_bits),dtype=np.uint8)
            progenitores[0] = poblacion[sorted[-1]]   # elite

            v = ventana
            for i in range(1,self.prog):
                progenitores[i] = poblacion[sorted[-self.rng.integers(0,v,1)]]
                v += ventana

            # 2) cruzar progenitores (cruza simple)
            poblacion[:self.prog] = progenitores
            for i in range(self.prog,self.pob):
                p1, p2 = self.rng.integers(0,self.prog,2)    # tomar progenitores al azar

                # punto de cruza: elegir inicio y fin del segmento a cruzar
                c1, c2 = np.sort(self.rng.integers(0, self.n_bits, 2))
                
                # cruzar: segmento del padre 1 y el resto del padre 2
                poblacion[i, :c1] = progenitores[p2, :c1]      # antes de c1
                poblacion[i, c1:c2] = progenitores[p1, c1:c2]  # c1 a c2
                poblacion[i, c2:] = progenitores[p2, c2:]      # de c2 al final

                # mutar
                if self.rng.random() < self.mutacion:
                    b = self.rng.integers(0,self.n_bits)
                    poblacion[i, b] ^= 1  # invierte 0 a 1 y viceversa


            # 3) evaluar fitness
            fit = self.calcular_fitness(poblacion)
            sorted = np.argsort(fit)    # indices que ordenan de menor a mayor
            if fit[sorted[-1]] > fit_elite:
                c_tol = 0
                fit_elite = fit[sorted[-1]]
                self.fit_history = np.vstack([self.fit_history, [g+1,fit_elite]])
            else:
                c_tol = c_tol + 1
                # condicion de parada
                if c_tol == self.tol:
                    self.fit_history = np.vstack([self.fit_history, [g+1,fit_elite]])
                    break

        return poblacion
