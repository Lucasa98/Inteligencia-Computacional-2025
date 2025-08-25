import numpy as np

class PerceptronMulticapa:
    def __init__(self, capas):
        self.capas = capas
        self.W = [] #pesos
        self.biases = [] #biases
        self.aprRate = 0.01 #tasa de aprendizaje

        #inicializar pesos y biases
        for i in range(len(capas) - 1):
            w = np.random.rand(capas[i], capas[i + 1]) #pesos entre capa i y capa i+1 
            bias = np.random.rand(capas[i + 1], 1 ) #bias para capa i+1
            self.W.append(w)
            self.biases.append(bias) 


    #funcion de activacion sigmoide   
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #derivada de la funcion sigmoide
    def sigmoid_deriv(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    

    #propagacion hacia adelante
    def forwardPropagation(self, x):
        a = x #activacion de la capa actual
        self.a = [a] #lista de activaciones por capa
        self.z = [] #lista de valores z por capa

        for W, biases in zip(self.W, self.biases): 
            z = np.dot(W, a)+ biases 
            a = self.sigmoid(z) 
            self.z.append(z) 
            self.a.append(a)
        return a
    

    #propagación hacia atrás
    def backPropagation(self, y):
        m = y.shape[0]
        deltas = [None]*len(self.W)

        #error en la capa de salida
        deltas[-1] = (self.a[-1] - y) * self.sigmoid_deriv(self.z[-1])

        #error en capas ocultas
        for l in range(len(deltas) - 2, -1, -1): #recorre de atrás hacia adelante
            deltas[l] = np.dot(self.W[l + 1].T, deltas[l + 1]) * self.sigmoid_deriv(self.z[l])

        #actualizar pesos y biases
        for l in range(len(self.W)):
            #aplicar descenso de gradiente
            self.W[l] -= self.aprRate * np.dot(deltas[l], self.a[l].T)/m #divide entre la cantidad de muestras
            #ajustar biases
            self.biases[l] -= self.aprRate * deltas[l]/m  

    def entrenar(self, x, y, epocas=1000):
        for epoca in range(epocas):
            for i in range(len(x)):
                self.forwardPropagation(x[i])
                self.backPropagation(y[i])


    def predecir(self, X):
        resultados = []
        for x in X:
            resultados.append(self.forwardPropagation(x))
        return resultados
