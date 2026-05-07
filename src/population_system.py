import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
import torch
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

def dxdt(t, x:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
    dx1 = a[0]*x[0]-a[1]*x[0]*x[1]
    dx2 = a[2]*x[1]+a[3]*x[0]*x[1] - a[4]*x[1]**2

    return torch.stack([dx1, dx2])

EPOCHS = 2000

# Temporal Grid
N_t = 100
t_range = (0, 1)
t_vals = np.linspace(t_range[0], t_range[1], N_t)
delta_t = t_vals[1] - t_vals[0]

a_vec = np.array([10, 5, 3, 1, 3])
x_0 = np.array([1.0, 0.5])
dtype = torch.float64

# print(f"True parameters: {a_vec}")

def rk2(t, x, a):
    k1 = dxdt(t, x, a)
    k2 = dxdt(t + delta_t/2, x + delta_t/2 * k1, a)
    return x + delta_t * k2

def integrate_rk2(x_0, a):
    x = x_0
    trajectory = [x_0]
    for t in t_vals[1:]:
        x = rk2(t, x, a)
        trajectory.append(x)
    return torch.stack(trajectory)

# Create a solution object by solving the system of ODEs
x0_torch = torch.tensor(x_0, dtype=dtype)
a_true = torch.tensor(a_vec, dtype=dtype)
solution = integrate_rk2(x0_torch, a_true).detach()
residual = lambda a: torch.sum((integrate_rk2(x0_torch, a) - solution)**2)

"""Plot of True Solution Trajectory"""
plt.figure(figsize=(10, 6))
plt.plot(solution[:, 0], solution[:, 1], "-o", label="Phase Trajectory", linewidth=5, markersize=8)
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Phase Plot of the Time Evolution of a Predator-Prey System")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "population_trajectory_example.svg")

a = torch.randn(5, dtype=dtype, requires_grad=True)
optimizer = torch.optim.Adam([a], lr=1e-2)

loss_history_tm_gd = []

for epoch in tqdm(range(EPOCHS), desc="Trajectory-Matching GD", colour="blue"):
    optimizer.zero_grad()

    loss = residual(a)
    loss.backward()
    optimizer.step()

    loss_history_tm_gd.append(loss.item())

# print(f"Trajectory-Matching GD estimated parameters:\n{a.detach().numpy()}")

"""Plotting Trajectory of the Trajectory-Matching GD solution"""
a_tm_gd = torch.tensor(a.detach().numpy(), dtype=dtype)
trajectory_tm_gd = integrate_rk2(x0_torch, a_tm_gd).detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(trajectory_tm_gd[:, 0], trajectory_tm_gd[:, 1], label="Trajectory-Matching GD", linewidth=5)
plt.plot(solution[:, 0], solution[:, 1], label="True Trajectory", linewidth=5, linestyle="dashed")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Trajectory of the Trajectory-Matching GD Solution")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectory_tm_gd.svg")

"""Now, do derivative matching GD"""
# Numerically compute the derivative data from the solution
derivative_data = torch.zeros_like(solution)
for i in range(1, len(t_vals)-1):
    derivative_data[i] = (solution[i+1] - solution[i-1]) / (2*delta_t)
derivative_data[0] = (solution[1] - solution[0]) / delta_t
derivative_data[-1] = (solution[-1] - solution[-2]) / delta_t

# derivative_matching_residual = lambda a: torch.norm(dxdt(t_vals, solution, a) - derivative_data)
def derivative_matching_residual(a):
    residuals = []
    for i in range(len(t_vals)):
        t = t_vals[i]
        x = solution[i]
        dxdt_pred = dxdt(t, x, a)
        residuals.append(dxdt_pred - derivative_data[i])
    return 0.5 * torch.sum(torch.stack(residuals)**2)

a = torch.randn(5, dtype=dtype, requires_grad=True)
optimizer = torch.optim.Adam([a], lr=1e-2)

loss_history_dm_gd = []

for epoch in tqdm(range(EPOCHS), desc="Derivative-Matching GD", colour="green"):
    optimizer.zero_grad()

    loss = derivative_matching_residual(a)
    loss.backward()
    optimizer.step()

    loss_history_dm_gd.append(loss.item())

# print(f"Derivative-Matching GD estimated parameters:\n{a.detach().numpy()}")

"""Plotting Trajectory of the Derivative-Matching GD solution"""
a_dm_gd = torch.tensor(a.detach().numpy(), dtype=dtype)
trajectory_dm_gd = integrate_rk2(x0_torch, a_dm_gd).detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(trajectory_dm_gd[:, 0], trajectory_dm_gd[:, 1], label="Derivative-Matching GD", linewidth=5)
plt.plot(solution[:, 0], solution[:, 1], "--o", label="True Trajectory", linewidth=5, markersize=8)
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Trajectory of the Derivative-Matching GD Solution")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectory_dm_gd.svg")

# # Plot the loss history
# plt.figure(figsize=(10, 6))

# plt.plot(loss_history_tm_gd, label="Loss Trajectory-Matching GD", linewidth=5)
# plt.plot(loss_history_dm_gd, label="Loss Derivative-Matching GD", linewidth=5)
# plt.yscale("log")

# plt.xlabel("Epoch")
# plt.ylabel(r"$\log$ Loss")

# plt.title("Comparison of Various Methods for Parameter Estimation in ODEs")

# plt.legend()
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "loss_history.svg")