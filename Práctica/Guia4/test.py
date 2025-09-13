import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import Som1D
from Som import Som

def cargarCSV(csvRelativePath, delimeter = ',') -> np.ndarray[any]:
    this_folder = os.path.abspath('') 
    fullPath = os.path.join(this_folder, csvRelativePath)
    return np.loadtxt(fullPath, dtype=float, delimiter=delimeter)

def plot_som_history(som: Som1D.Som1D, data: np.ndarray[float]):
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(data[:,0], data[:,1])
    scat, = ax.plot([], [], 'o-', lw=2)  # neurons with lines

    # define limits from all history
    all_weights = np.vstack(som.history)
    ax.set_xlim(all_weights[:,0].min()-0.2, all_weights[:,0].max()+0.2)
    ax.set_ylim(all_weights[:,1].min()-0.2, all_weights[:,1].max()+0.2)
    ax.set_aspect("equal")

    def update(frame):
        W = som.history[frame]
        scat.set_data(W[:,0], W[:,1])
        return scat,

    ani = FuncAnimation(fig, update, frames=len(som.history), interval=50, blit=True)
    plt.show()

def animate_som2d(som: Som, data: np.ndarray[float]):
    N = som.W.shape[0]
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(data[:,0], data[:,1])

    # set axis limits based on all history
    all_weights = np.vstack([W.reshape(-1,2) for W in som.history])
    ax.set_xlim(all_weights[:,0].min()-0.2, all_weights[:,0].max()+0.2)
    ax.set_ylim(all_weights[:,1].min()-0.2, all_weights[:,1].max()+0.2)
    ax.set_aspect("equal")

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
        if frame >= len(som.history):
            return
        W = som.history[frame]                      # (N, N, 2)
        scat.set_offsets(np.c_[W.reshape(-1, 2)])   # (N*N, 2)
        for i in range(N):
            row_lines[i].set_data(W[i,:,0], W[i,:,1])   # rows
            col_lines[i].set_data(W[:,i,0], W[:,i,1])   # cols
        return [scat] + row_lines + col_lines

    ani = FuncAnimation(fig, update, frames=len(som.history)+10, init_func=init, interval=16)
    plt.show()

data = np.loadtxt('./te.csv', dtype=float, delimiter=',')
som1d = Som1D.Som1D(32,0.2)
som1d.entrenar(data, 50)

som2d = Som(10,[(0.9, 0.7),(0.7, 0.1),(0.1, 0.01)], [30, 300, 50])
som2d.entrenar(data)

animate_som2d(som2d, data)
#plot_som_history(som1d, data)