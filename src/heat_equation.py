import numpy as np
from scipy.sparse import csr_matrix, kron, eye

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
x_vals = np.linspace(x_range[0], x_range[1], N)
y_vals = np.linspace(y_range[0], y_range[1], N)
X,Y = np.meshgrid(x_vals, y_vals)
h = X[0,1] - X[0,0]

# Generate Heat Equation Solution
alpha = 0.25
forcing_fxn = lambda x,y,t: (2 * alpha - 1)*np.cos(x)*np.sin(y)*np.exp(-t)
u_exact = lambda x,y,t: np.cos(x)*np.sin(y)*np.exp(-t)

"""Plot Solution at t=0, and t=2pi"""
plt.figure(figsize=(10, 5))

# Plot at t=0
plt.subplot(1, 2, 1)
plt.contourf(X, Y, u_exact(X, Y, t_range[0]), levels=20, cmap='icefire', vmin=-1, vmax=1)
plt.colorbar()
plt.title(f'Solution at t={t_range[0]:.2f}')
plt.xlabel('x')
plt.ylabel('y')

# Plot at last time point
plt.subplot(1, 2, 2)
plt.contourf(X, Y, u_exact(X, Y, t_range[1]), levels=20, cmap='icefire', vmin=-1, vmax=1)
plt.colorbar()
plt.title(f'Solution at t={t_range[1]:.2f}')
plt.xlabel('x')
plt.ylabel('y')

plt.suptitle('Heat Equation Solution with Forcing Term', fontsize=16)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "HeatEquation_Solution.svg")