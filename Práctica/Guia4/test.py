import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import Som1D

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

data = np.array([
    [0.9,0.9],
    [0.9,-0.9],
    [0.9,1.1],
    [0.9,-1.1],
    [1.1,0.9],
    [1.1,-0.9],
    [1.1,1.1],
    [1.1,-1.1],
    [-0.9,0.9],
    [-0.9,-0.9],
    [-0.9,1.1],
    [-0.9,-1.1],
    [-1.1,0.9],
    [-1.1,-0.9],
    [-1.1,1.1],
    [-1.1,-1.1],
])
som = Som1D.Som1D(32,0.2)
som.entrenar(data, 50)
plot_som_history(som, data)