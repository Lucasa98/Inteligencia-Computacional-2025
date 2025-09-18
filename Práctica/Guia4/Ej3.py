import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Kmeans import Kmeans
from sklearn.datasets import load_iris

# cargar dataset
data = load_iris()
x = data.data
y = data.target

# solo 2D para visualización
x_2d = x[:, 2:4]

# número de clusters
k = 10

# entrenar K-means
kmeans = Kmeans(N=k)
kmeans.entrenar(x)
y_pred = kmeans.predecir(x)

# crear figura
fig, ax = plt.subplots(figsize=(6,5))
scatter = ax.scatter(x_2d[:,0], x_2d[:,1], c=y_pred, cmap='tab10', s=50, alpha=0.7)

# scatter de centroides (vacío al inicio)
centroid_plot = ax.scatter([], [], s=200, marker='X', edgecolor='black', zorder=5)

ax.set_xlabel('Largo de Pétalo')
ax.set_ylabel('Ancho de Pétalo')
ax.set_title(f'K-means Animación (k={k})')
ax.grid(True)

def init():
    centroid_plot.set_offsets(np.empty((0, 2)))  # array vacío 2D
    return [centroid_plot]

def update(frame):
    centroids_2d = kmeans.history[frame][:, 2:4]
    centroid_plot.set_offsets(centroids_2d)
    centroid_plot.set_array(np.arange(k))  # asignar colores a cada centroide
    ax.set_title(f'Iteración {frame+1}/{len(kmeans.history)}')
    return [centroid_plot]

ani = FuncAnimation(fig, update, frames=len(kmeans.history), init_func=init, interval=200, repeat=False)
plt.show()
