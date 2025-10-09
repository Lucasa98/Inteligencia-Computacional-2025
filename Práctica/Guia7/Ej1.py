import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PSO import PSO

# Función objetivo
def f1(x):
    x = np.atleast_1d(x)
    f = -x*np.sin(np.sqrt(np.abs(x)))
    return f if f.shape[0] > 1 else f[0]

# Crear PSO y ejecutar
pso = PSO(func=f1, n_particulas=5, dim=1, xmin=-512, xmax=512, max_it=200, it_tol=50)
x_best, y_best = pso.actualizar()

# Preparar gráfico
X = np.linspace(-512, 512, 1000)
Y = f1(X)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(X, Y, color='lightgray')

# Colores aleatorios para partículas
colors = np.random.rand(pso.n_particulas, 3)

# Inicializar scatter con posiciones dummy para evitar errores
dummy_pos = np.zeros((pso.n_particulas, 2))
scat = ax.scatter(dummy_pos[:,0], dummy_pos[:,1], color=colors, s=50)

# Mejor global con X roja
best_pt = ax.scatter([], [], color='red', marker='x', s=100, label='Mejor global')

frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlim(-520, 520)
ax.set_ylim(np.min(Y)-10, np.max(Y)+10)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Animación de PSO')
ax.grid(True)
ax.legend()

# Historial sin interpolar
historial = np.array(pso.historial)  # (it, n_particulas, dim)
best_global_hist = np.array(pso.best_global_hist)

total_frames = len(historial)

def update(frame):
    posiciones = historial[frame]
    scat.set_offsets(np.c_[posiciones[:,0], f1(posiciones[:,0])])
    best = best_global_hist[frame]
    best_pt.set_offsets([best[0], f1(best[0])])
    frame_text.set_text(f'Frame {frame+1}/{total_frames}')
    return scat, best_pt, frame_text

ani = FuncAnimation(fig, update, frames=total_frames, interval=700, blit=True)
plt.show()
