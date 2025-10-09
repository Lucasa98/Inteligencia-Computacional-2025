import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PSO import PSO

# === Función objetivo ===
def f1(x):
    x = np.atleast_1d(x)
    f = -x * np.sin(np.sqrt(np.abs(x)))
    return f if f.shape[0] > 1 else f[0]

# === Crear y ejecutar PSO ===
pso = PSO(func=f1, n_particulas=5, dim=1, xmin=-512, xmax=512, max_it=200, it_tol=50)
x_best, y_best = pso.actualizar()

# === Parámetros de animación ===
interpolate = False
subframes = 10

# === Preparar gráfico ===
X = np.linspace(-512, 512, 1000)
Y = f1(X)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(X, Y, color='lightgray')

colors = np.random.rand(pso.n_particulas, 3)
dummy_pos = np.zeros((pso.n_particulas, 2))
scat = ax.scatter(dummy_pos[:,0], dummy_pos[:,1], color=colors, s=50)
lines = [ax.plot([], [], color=colors[i], lw=1.5, alpha=0.7)[0] for i in range(pso.n_particulas)]
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

# === Historial ===
historial = np.array(pso.historial)        # (it, n_particulas, dim)
best_global_hist = np.array(pso.best_global_hist)
total_iters = len(historial)

# === Generar frames ===
if interpolate:
    interp_positions = []
    interp_best = []
    segment_starts = []
    segment_ends = []

    for i in range(total_iters - 1):
        start_x = historial[i][:, 0]
        end_x   = historial[i+1][:, 0]
        start_xy = np.stack([start_x, f1(start_x)], axis=1)
        end_xy   = np.stack([end_x,   f1(end_x)], axis=1)

        best_xy = np.array([best_global_hist[i][0], f1(best_global_hist[i][0])])  # sin interpolar

        for t in np.linspace(0, 1, subframes, endpoint=False):
            interp_positions.append(start_xy + t * (end_xy - start_xy))
            interp_best.append(best_xy)  # mismo valor toda la subinterpolación
            segment_starts.append(start_xy)
            segment_ends.append(end_xy)

    # última iteración
    last_xy = np.stack([historial[-1][:,0], f1(historial[-1][:,0])], axis=1)
    last_best_xy = np.array([best_global_hist[-1][0], f1(best_global_hist[-1][0])])
    interp_positions.append(last_xy)
    interp_best.append(last_best_xy)
    segment_starts.append(last_xy)
    segment_ends.append(last_xy)

    interp_positions = np.array(interp_positions)
    interp_best = np.array(interp_best)
    segment_starts = np.array(segment_starts)
    segment_ends = np.array(segment_ends)
    total_frames = len(interp_positions)

else:
    # sin interpolación: un frame por iteración
    total_frames = total_iters
    interp_positions = np.stack([np.stack([historial[i][:,0], f1(historial[i][:,0])], axis=1)
                                 for i in range(total_iters)])
    interp_best = np.stack([[best_global_hist[i][0], f1(best_global_hist[i][0])] for i in range(total_iters)])
    segment_starts = interp_positions.copy()
    segment_ends = interp_positions.copy()

# === Función update ===
def update(frame):
    pos = interp_positions[frame]
    scat.set_offsets(pos)

    # Dibujar rectas a destino solo si interpolate=True
    if interpolate:
        for i, line in enumerate(lines):
            start = segment_starts[frame][i]
            end = segment_ends[frame][i]
            line.set_data([start[0], end[0]], [start[1], end[1]])
    else:
        for line in lines:
            line.set_data([], [])

    # Mejor global (no se interpola nunca)
    best_xy = np.atleast_2d(interp_best[frame])
    best_pt.set_offsets(best_xy)

    frame_text.set_text(f'Frame {frame+1}/{total_frames}')
    return [scat, best_pt, frame_text, *lines]

ani = FuncAnimation(fig, update, frames=total_frames, interval=(70 if interpolate else 300), blit=True)
plt.show()
