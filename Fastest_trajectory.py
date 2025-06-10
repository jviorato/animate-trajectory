import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# Device setup
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

# Obstacle loss components
def circle_obstacle_loss(V_mat, center, radius, lam=1.0):
    """Penalize proximity to a circle defined by center (x, y) and radius."""
    xs = pt.stack([v[0] for v in V_mat])
    ys = pt.stack([v[1] for v in V_mat])
    dists_sq = (xs - center[0])**2 + (ys - center[1])**2
    inside = pt.clamp(radius**2 - dists_sq, min=0.0)
    return lam * pt.sum(inside)

def rectangle_obstacle_loss(V_mat, corners, lam=1.0):
    """Penalize being inside a rectangle defined by 4 corners (x, y)."""
    xs = pt.stack([v[0] for v in V_mat])
    ys = pt.stack([v[1] for v in V_mat])
    x_vals = [pt.tensor(c[0], device=xs.device) for c in corners]
    y_vals = [pt.tensor(c[1], device=ys.device) for c in corners]
    x_min = pt.min(pt.stack(x_vals))
    x_max = pt.max(pt.stack(x_vals))
    y_min = pt.min(pt.stack(y_vals))
    y_max = pt.max(pt.stack(y_vals))

    inside_x = (xs > x_min) & (xs < x_max)
    inside_y = (ys > y_min) & (ys < y_max)
    inside = inside_x & inside_y
    return lam * pt.sum(inside.float())

def triangle_obstacle_loss(V_mat, triangle_pts, lam=1.0):
    """Penalize points inside triangle defined by 3 (x,y) vertices."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
               (p2[0] - p3[0]) * (p1[1] - p3[1])

    xys = pt.stack([pt.stack((v[0], v[1])) for v in V_mat])
    p1, p2, p3 = [pt.tensor(p, device=xys.device) for p in triangle_pts]

    b1 = sign(xys.T, p1, p2) < 0.0
    b2 = sign(xys.T, p2, p3) < 0.0
    b3 = sign(xys.T, p3, p1) < 0.0

    inside = ((b1 == b2) & (b2 == b3)).float()
    return lam * pt.sum(inside)


# Dynamics definition
def v_prime_components_autograd(v, dt):
    dt2 = dt * dt / 2
    x, y, a, s, w = v
    _0 = pt.zeros_like(x)

    A1 = pt.stack([x, y, a, s, w])
    A2 = pt.stack([s * pt.cos(a), s * pt.sin(a), w, _0, _0])
    A3 = pt.stack([-s * w * pt.sin(a), s * w * pt.cos(a), _0, _0, _0])
    A = A1 + dt * A2 + dt2 * A3

    B2 = pt.stack([_0, _0, _0, pt.ones_like(x), _0])
    B3 = pt.stack([pt.cos(a), pt.sin(a), _0, _0, _0])
    B = dt * B2 + dt2 * B3

    C2 = pt.stack([_0, _0, _0, _0, pt.ones_like(x)])
    C3 = pt.stack([_0, _0, pt.ones_like(x), _0, _0])
    C = dt * C2 + dt2 * C3
    return A, B, C

def compute_path_autograd(v0, F_mat, t, target):
    V_mat = [v0]
    for i in range(len(t) - 1):
        A, B, C = v_prime_components_autograd(V_mat[i], t[i + 1] - t[i])
        V_mat.append(A + F_mat[i, 0] * B + F_mat[i, 1] * C)
    loss = (1 / 5) * pt.sum((V_mat[-1] - target) ** 2)

    # Add obstacle penalties
    loss += circle_obstacle_loss(V_mat, center=(2.5, 2.5), radius=1.5, lam=10.0)
    loss += rectangle_obstacle_loss(V_mat, corners=[(-1,0), (-1,2.5), (4, 2.5), (4,0)], lam=10.0)
    # loss += triangle_obstacle_loss(V_mat, triangle_pts=[(0,0), (2,4), (4,1)], lam=10.0)

    return V_mat, loss

# Initial and target states
v0 = pt.tensor([0, 0, 1.57079632679489, .5, 0], dtype=pt.float32, device=device)
target = pt.tensor([5, 1, 1.57079632679489, 2, 0], dtype=pt.float32, device=device)

# Optimization config
min_steps, max_steps = 8, 200
dt_max = 0.1
loss_threshold = 1e-5
best_result = None

for n_ts in range(min_steps, max_steps):
    ts = pt.linspace(0, n_ts * dt_max, n_ts + 1, device=device)
    F_mat = pt.nn.Parameter(pt.zeros((n_ts, 2), dtype=pt.float32, device=device))
    optimizer = pt.optim.Adam([F_mat], lr=0.08)

    all_trajectories = [] 
    all_forcings = []

    for i in range(300):
        optimizer.zero_grad()
        V_mat, loss = compute_path_autograd(v0, F_mat, ts, target)
        loss.backward()
        optimizer.step()
        with pt.no_grad():
            F_mat.clamp_(min=-0.8, max=0.8)

        if i % 10 == 9:  # Save every 10 steps
            all_trajectories.append(pt.stack(V_mat).detach().cpu().numpy())
            all_forcings.append(F_mat.detach().cpu().clone().numpy())

    if loss.item() < loss_threshold:
        best_result = (ts, V_mat, F_mat.detach().cpu(), loss.item(), all_trajectories)
        break

# Arrow direction function
def arrowDir(V):
    return V[3] * np.cos(V[2]), V[3] * np.sin(V[2])

# Final Plotting
if best_result:
    ts, V_mat, F_mat, final_loss, all_trajectories = best_result

    # convert to NumPy
    V_np = pt.stack(V_mat).detach().cpu().numpy()
    ts_np = ts.detach().cpu().numpy()

 # save for animation
    np.savez('trajectory_data.npz', V_np=V_np, ts=ts_np)
    print("👉 Trajectory data saved to trajectory_data.npz")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Plot all intermediate (learning) trajectories
    for traj in all_trajectories:
        ax1.plot(traj[:, 0], traj[:, 1], lw=1.0, color='dodgerblue', alpha=0.15)

    # Plot final trajectory
    ax1.plot(V_np[:, 0], V_np[:, 1], lw=2, color='dodgerblue', label='Trajectory')
    ax1.scatter(V_np[0, 0], V_np[0, 1], marker='*', color='sandybrown', s=200, label='Start')
    ax1.scatter(target[0].cpu(), target[1].cpu(), marker='s', color='sandybrown', s=120, label='Target')
    ax1.scatter(V_np[-1, 0], V_np[-1, 1], marker='o', color='dodgerblue', s=60, label='End')

    ax1.arrow(*V_np[0, 0:2], *arrowDir(V_np[0]), color='sandybrown', head_width=0.15, head_length=0.15, lw=4, zorder=2)
    ax1.arrow(*target[0:2].cpu().numpy(), *arrowDir(target.cpu().numpy()), color='sandybrown', head_width=0.15, head_length=0.15, lw=3, zorder=2)
    ax1.arrow(*V_np[-1, 0:2], *arrowDir(V_np[-1]), color='dodgerblue', head_width=0.1, head_length=0.1, lw=2, zorder=3)

    ax1.set_title("Fastest Trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    # Control inputs
    ax2.plot(ts[:-1].cpu(), F_mat[:, 0], label='Acceleration φ', color='deeppink')
    ax2.plot(ts[:-1].cpu(), F_mat[:, 1], label='Turning ψ', color='limegreen')
    ax2.set_title("Control Inputs")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Force")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    print(f"Found solution in {len(ts)-1} steps with final loss {final_loss:.4e}")
else:
    print("No solution found within max steps.")
