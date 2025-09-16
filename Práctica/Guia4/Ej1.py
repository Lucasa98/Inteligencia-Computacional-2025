import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Som import Som
from Som1D import Som1D

# ///////////////////// Funciones auxiliares /////////////////////
def cargarCSV(csvRelativePath, delimeter = ',') -> np.ndarray[any]:
    this_folder = os.path.abspath('') 
    fullPath = os.path.join(this_folder, csvRelativePath)
    return np.loadtxt(fullPath, dtype=float, delimiter=delimeter)

def animate_som2d(som: Som, data: np.ndarray[float]):
    N = som.W.shape[0]
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(data[:,0], data[:,1])

    # set axis limits based on all history
    all_weights = np.vstack([W.reshape(-1,2) for W in som.history])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

    # titulo
    title = ax.set_title("epoca 0")

    # scatter all neurons
    scat = ax.scatter(som.history[0][:,:,0], som.history[0][:,:,1], c="blue")

    # connect horizontal and vertical neighbors
    row_lines = [None for _ in range(N)]
    col_lines = [None for _ in range(N)]
    for i in range(N):
        row_lines[i], = ax.plot(som.history[0][i,:,0], som.history[0][i,:,1], c="gray")   # rows
        col_lines[i], = ax.plot(som.history[0][:,i,0], som.history[0][:,i,1], c="gray")   # cols

    def init(): # esto para que ande en linux
        return [scat] + row_lines + col_lines

    def update(frame):
        title.set_text(f"epoca {frame}")
        if frame >= len(som.history):
            return
        W = som.history[frame]                      # (N, N, 2)
        scat.set_offsets(np.c_[W.reshape(-1, 2)])   # (N*N, 2)
        for i in range(N):
            row_lines[i].set_data(W[i,:,0], W[i,:,1])   # rows
            col_lines[i].set_data(W[:,i,0], W[:,i,1])   # cols
        return [scat] + row_lines + col_lines

    ani = FuncAnimation(fig, update, frames=len(som.history)+10, init_func=init, interval=64)
    plt.show()
# ////////////////////////////////////////////////////////////////

data = cargarCSV('Práctica\\Guia4\\data\\te.csv')

som2d10 = Som(
    N=10,
    vecindades=[(5,5), (4,1), (0,0)],
    tasa_aprendizaje=[(0.9, 0.7), (0.7, 0.1), (0.1, 0.01)],
    epocas=[100, 200, 600]
)
som2d5 = Som(
    N=5,
    vecindades=[(3,3), (2,1), (0,0)],
    tasa_aprendizaje=[(0.9, 0.7), (0.7, 0.1), (0.1, 0.01)],
    epocas=[100, 200, 600]
)
som2d2 = Som(
    N=2,
    vecindades=[(3,3), (2,1), (0,0)],
    tasa_aprendizaje=[(0.9, 0.7), (0.7, 0.1), (0.1, 0.01)],
    epocas=[100, 200, 600]
)

som2d10.entrenar(data)
animate_som2d(som2d10, data)




# Para 1D

def animate_som1d(som: Som1D, data: np.ndarray[float]):
    N = som.W.shape[0]
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(data[:,0], data[:,1], marker="x", c="gray")

    all_weights = np.vstack([W for W in som.history])
    ax.set_xlim(all_weights[:,0].min()-0.1, all_weights[:,0].max()+0.1)
    ax.set_ylim(all_weights[:,1].min()-0.1, all_weights[:,1].max()+0.1)
    ax.set_aspect("equal")

    # título dinámico
    title = ax.set_title("época 0")

    # scatter de neuronas
    scat = ax.scatter(som.history[0][:,0], som.history[0][:,1], c="blue")

    # línea conectando neuronas
    line, = ax.plot(som.history[0][:,0], som.history[0][:,1], 'o-', c="red")

    def init():
        return [scat, line]

    def update(frame):
        if frame >= len(som.history):
            return
        W = som.history[frame]   # (N, entradas)
        scat.set_offsets(W)      # actualizar posiciones de neuronas
        line.set_data(W[:,0], W[:,1])  # conectar neuronas en orden
        title.set_text(f"época {frame}")
        return [scat, line]

    ani = FuncAnimation(fig, update, frames=len(som.history), init_func=init, interval=64)
    plt.show()


data = cargarCSV('Práctica\\Guia4\\data\\te.csv')

som1d25 = Som1D(
    N=25,
    vecindades=[(10,10), (9,1), (0,0)],
    tasa_aprendizaje=[(0.9, 0.7), (0.7, 0.1), (0.1, 0.01)],
    epocas=[100, 200, 600]
)

som1d25.entrenar(data)
animate_som1d(som1d25, data)
