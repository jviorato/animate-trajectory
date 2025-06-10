import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Load precomputed trajectory
data = np.load('trajectory_data.npz')
V_np = data['V_np']   # shape (N,5): x, y, θ, s, w
ts   = data['ts']     # shape (N,)

# 2. Set up plot
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(V_np[:,0].min() - 1, V_np[:,0].max() + 1)
ax.set_ylim(V_np[:,1].min() - 1, V_np[:,1].max() + 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Car Trajectory Animation')

# 3. Initialize line and “car” marker
path_line, = ax.plot([], [], linestyle='--', linewidth=1, alpha=0.7)
car_marker, = ax.plot([], [], marker=(3, 0, 0), markersize=15, linestyle='None')

def init():
    path_line.set_data([], [])
    car_marker.set_data([], [])
    return path_line, car_marker

def update(frame):
    x, y, theta = V_np[frame, 0], V_np[frame, 1], V_np[frame, 2]
    # update path so far
    path_line.set_data(V_np[:frame+1, 0], V_np[:frame+1, 1])
    # update car position & rotation
    car_marker.set_data(x, y)
    car_marker.set_marker((3, 0, np.degrees(theta)))
    return path_line, car_marker

# 4. Create the animation
# interval=ts spacing converted to milliseconds? If ts is in seconds:
#    interval = np.diff(ts, prepend=ts[0]) * 1000
# but here we’ll just fix a frame rate:
ani = FuncAnimation(fig, update, frames=len(V_np),
                    init_func=init, blit=True, interval=50)

plt.tight_layout()
plt.show()

# (Optional) to save as MP4:
# ani.save('trajectory.mp4', fps=20, dpi=150)
