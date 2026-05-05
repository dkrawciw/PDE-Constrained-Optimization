import torch
import numpy as np
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import splu

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Temporal Grid
N_t = 1000
t_range = (0, 1)
delta_t = (t_range[1] - t_range[0]) / N_t

# Spacial Grid
N = 64
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], N, endpoint=False)
y_vals = np.linspace(y_range[0], y_range[1], N, endpoint=False)
X,Y = np.meshgrid(x_vals, y_vals)
h = X[0,1] - X[0,0]

# Create the periodic Laplacian operator
D2 = np.zeros((N,N))
D2 += np.eye(N, k=0) * -2
D2 += np.eye(N, k=1)
D2 += np.eye(N, k=-1)
D2[0, -1] = 1
D2[-1, 0] = 1
D2 /= (h**2)
D2 = csr_matrix(D2)
L = kron(eye(N, format="csr"), D2, format="csr") + kron(D2, eye(N, format="csr"), format="csr")

# Create the periodic gradient operators
D1 = np.zeros((N,N))
D1 += np.eye(N, k=1)
D1 += -np.eye(N, k=-1)
D1[0, -1] = -1
D1[-1, 0] = 1
D1 /= (2*h)
D1 = csr_matrix(D1)

grad_x = kron(eye(N, format="csr"), D1, format="csr")
grad_y = kron(D1, eye(N, format="csr"), format="csr")

v = lambda x,y,t: np.array([2,0])

def advection_operator(t):
    """Create the matrix for -v dot grad at time t."""
    velocity = v(X, Y, t)

    if np.shape(velocity) == (2,):
        velocity_x = np.full_like(X, velocity[0], dtype=float)
        velocity_y = np.full_like(Y, velocity[1], dtype=float)
    else:
        velocity_x, velocity_y = velocity

    return -(diags(velocity_x.ravel()) @ grad_x + diags(velocity_y.ravel()) @ grad_y)

soln = np.zeros((N, N, N_t+1))
# soln[:,:,0] = np.cos(X)*np.sin(Y)
soln[:,:,0] = np.exp(-(X**2 + (Y-np.pi)**2))
diffusion_coefficient = 0.25

u = soln[:,:,0].ravel()
I = eye(N**2, format="csr")

M = diffusion_coefficient*L + advection_operator(0)
LHS = splu((I - delta_t/2*M).tocsc())

for i in tqdm(range(N_t), desc="Solving advection-diffusion equation"):
    # t = t_range[0] + i*delta_t
    # M = diffusion_coefficient*L + advection_operator(t + delta_t/2)
    RHS = (I + delta_t/2*M) @ u
    
    u = LHS.solve(RHS)

    soln[:,:,i+1] = u.reshape(N,N)

# Plot the solution at the initial and final time steps
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
vmin = min(soln[:, :, 0].min(), soln[:, :, -1].min())
vmax = max(soln[:, :, 0].max(), soln[:, :, -1].max())

initial_plot = axes[0].contourf(
    X,
    Y,
    soln[:, :, 0],
    levels=50,
    cmap="icefire",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title(f"Initial condition at t={t_range[0]:.2f}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

final_plot = axes[1].contourf(
    X,
    Y,
    soln[:, :, -1],
    levels=50,
    cmap="icefire",
    vmin=vmin,
    vmax=vmax,
)
axes[1].set_title(f"Final condition at t={t_range[1]:.2f}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

fig.suptitle("Periodic Advection-Diffusion Equation Solution\nGaussian Travelling to the Right", fontsize=16)

fig.colorbar(final_plot, ax=axes, label="Concentration")
plt.savefig(OUTPUT_DIR / "advection_diffusion.svg")