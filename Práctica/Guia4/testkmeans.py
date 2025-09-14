import numpy as np
import matplotlib.pyplot as plt
from Kmeans import Kmeans

rng = np.random.default_rng()

# generar datos
grupo1 = rng.uniform(-0.2,0.2,(30, 2)) + np.array([-1,1])
grupo2 = rng.uniform(-0.2,0.2,(30, 2)) + np.array([-0.5,-0.5])
grupo3 = rng.uniform(-0.2,0.2,(30, 2)) + np.array([0.5,0.5])
grupo4 = rng.uniform(-0.2,0.2,(30, 2)) + np.array([1,-1])
data = np.concatenate([grupo1, grupo2, grupo3, grupo4])

kmean = Kmeans(4)
kmean.entrenar(data)

# graficar
fig, ax = plt.subplots(1,1)
ax.grid()

ax.scatter(data[:,0], data[:,1], c='b')
ax.scatter(kmean.centroides[:,0], kmean.centroides[:,1], c='r')
plt.show()