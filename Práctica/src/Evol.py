import numpy as np
from typing import Callable

class Evol:
    def __init__(self,
                 fit_fun: Callable[[np.ndarray[np.uint8]], np.ndarray[float]],
                 encode_fun: Callable[[np.ndarray[float]], np.ndarray[np.uint8]],
                 decode_fun: Callable[[np.ndarray[np.uint8]], np.ndarray[float]],
                 max_gen: int = 20,
                 poblacion: int = 100,
                 progenitores: int = 10,
                 precision: int = 8,
                 mutacion: float = 0.01,
                 tolerancia: int = 5):
        """
        Args:
            fit_fun (Callable[[np.ndarray[np.uint8]], np.ndarray[float]]): funcion para calcular fitness
            encode_fun (Callable[[np.ndarray[float]], np.ndarray[np.uint8]]): funcion para codificar
            decode_fun (Callable[[np.ndarray[np.uint8]], np.ndarray[float]]): funcion para decodificar
            max_gen (int, optional): numero maximo de generaciones. Defaults to 20.
            poblacion (int, optional): poblacion por generacion. Defaults to 100.
            progenitores (int, optional): progenitores elegidos en cada generacion. Defaults to 10.
            precision (int, optional): numero de bits del cromosoma. Defaults to 8.
            mutacion (float, optional): probabilidad de mutacion. Defaults to 0.01.
        """
        self.max_gen = max_gen
        self.pob = poblacion
        self.prog = progenitores
        self.calcular_fitness = fit_fun
        self.encode = encode_fun
        self.decode = decode_fun
        self.mutacion = mutacion
        self.tol = tolerancia
        self.n_bits = precision
        self.rng = np.random.default_rng()

    def evolve(self) -> np.ndarray[float]:
        # 1) inicializar la poblacion al azar
        poblacion = self.rng.integers(0,2,(self.pob,self.n_bits),dtype=np.uint8)
        ventana = self.pob//self.prog

        # 2) calcular fitness
        fit = self.calcular_fitness(poblacion)
        sorted = np.argsort(fit)    # indices que ordenan de menor a mayor

        c_tol: int = 0   # contador de generaciones sin mejora
        fit_elite = fit[sorted[-1]]
        for _ in range(self.max_gen):
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
                c = self.rng.integers(0,self.n_bits,2).sort()  # punto de cruza
                poblacion[i,:c] = progenitores[p1][:c]
                poblacion[i,c:] = progenitores[p2][c:]
                # mutar
                if self.rng.random() < self.mutacion:
                    b = self.rng.integers(0,self.n_bits)
                    poblacion[i][b] = 0 if poblacion[i][b] == 1 else 1

            # 3) evaluar fitness
            fit = self.calcular_fitness(poblacion)
            sorted = np.argsort(fit)    # indices que ordenan de menor a mayor
            if fit[sorted[-1]] > fit_elite:
                c_tol = 0
                fit_elite = fit[sorted[-1]]
            else:
                c_tol = c_tol + 1
                # condicion de parada
                if c_tol == self.tol:
                    break

        return poblacion
