import numpy as np  

class PSO:
    def __init__(self, func, n_particulas, dim, xmin, xmax, max_it, it_tol):
        self.func = func
        self.n_particulas = n_particulas
        self.dim = dim
        self.xmin = xmin
        self.xmax = xmax
        self.max_it = max_it
        self.it_tol = it_tol
        self.it = 0

        #INICIALIZACIÓN
        self.x = self.xmin + np.random.rand(self.n_particulas, self.dim) * (self.xmax - self.xmin) # Posiciones iniciales
        self.v = np.zeros((self.n_particulas, self.dim))  # Velocidades iniciales
        self.x_best_local = self.x.copy()  # Mejores posiciones individuales
        y = np.array([self.func(self.x[i,:]) for i in range(self.n_particulas)])
        best_idx = np.argmin(y)
        self.x_best_global = self.x[best_idx, :].reshape(1, self.dim)  # Mejor posición global


    def actualizar(self):

        c_it = 0

        for it in range(self.max_it):

            for p in range(self.n_particulas):

                #sMEJOR POSICIÓN INDIVIDUAL
                #comparar la función objetivo evaluada en la particula con la función objetivo evaluada en la mejor posición histórica de esa partícula
                if self.func(self.x[p,:]) < self.func(self.x_best_local[p,:]):
                    self.x_best_local[p,:] = self.x[p,:] 
                    

                #MEJOR POSICIÓN GLOBAL
                #comparar la función de objetivo evaluada en la mejor posición de la partícula con la función evaluada en la mejor posición histórica del emjambre
                if self.func(self.x_best_local[p,:]) < self.func(self.x_best_global):
                    self.x_best_global = self.x_best_local[p,:].reshape(1, self.dim)   
                    c_it+=1

                #condición de corte por tolerancia
                if c_it >= self.it_tol:
                    print(f'Convergencia alcanzada en la iteración {it}')
                    return self.x_best_global, self.func(self.x_best_global)

            
            #FUNCIONES ESTOCÁSTICAS
            r1 = np.random.rand(self.n_particulas, self.dim)
            r2 = np.random.rand(self.n_particulas, self.dim)

            #FUNCIONES PARA LOS COEFICIENTES
            cmax = 2.5
            cmin = 0.5
            c1 = cmax - (cmax - cmin)*(self.it/self.max_it) #decreciente
            c2 = cmin + (cmax - cmin)*(self.it/self.max_it) #creciente

            for p in range(self.n_particulas):  

                #ACTUALIZAR VELOCIDAD
                experiencia_personal = c1*r1[p,:]*(self.x_best_local[p,:]-self.x[p,:])
                experiencia_enjambre = c2*r2[p,:]*(self.x_best_global - self.x[p,:])
                self.v[p,:] = self.v[p,:] + experiencia_personal + experiencia_enjambre

                #ACTUALIZAR POSICIÓN
                self.x[p,:] += self.v[p,:] 

                #limitar a los rangos
                self.x[p,:] = np.clip(self.x[p,:], self.xmin, self.xmax)
            
            self.it += 1




