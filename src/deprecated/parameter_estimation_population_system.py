import numpy as np
import torch

from PopulationSystem import PopulationSystem
from PeriodicAdvectionDiffusion import PeriodicAdvectionDiffusion

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle as pkl

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

TRIALS_PER_FRAME_COUNT = 40
EPOCHS = 20
CASCADE_NUM_BASIS = 20
CASCADE_LAMBDA_ODE = 1.0
CASCADE_LR = 1e-2

dtype = torch.float64

"""Optimize parameters to match the solution trajectory"""
x0 = torch.rand(2, dtype=dtype)
a_true = torch.randn(5, dtype=dtype)
a_true = torch.nn.functional.softplus(a_true)  # ensure positivity of parameters

solver = PopulationSystem()
with torch.no_grad():
    solution = solver.solve(
        x0,
        a_true,
    )

a_start = a_true + torch.randn(5, dtype=dtype)
a_start_list = []
for _ in range(TRIALS_PER_FRAME_COUNT):
    a_start = a_true + torch.randn(5, dtype=dtype)*0.5
    a_start_list.append(a_start)

last_residuals = []
num_frames_list = range(5, solution.shape[0], 10)

# residual_samples = []

# for num_frames in tqdm(num_frames_list, desc="Varying # of Frames Trained Against", colour="green"):

#     residual_trial = []
#     for i in tqdm(range(TRIALS_PER_FRAME_COUNT), desc="Trials", colour="blue"):
#         frames = torch.linspace(0, solution.shape[0] - 1, num_frames).round().long()
#         residual_trajectory_matching = lambda a: torch.mean((solver.solve(x0, a)[frames,:] - solution[frames,:])**2)

#         a = a_start_list[i].clone().detach().requires_grad_(True)
#         # a.requires_grad_(True)
#         optimizer = torch.optim.Adam([a], lr=1e-2)

#         for epoch in range(EPOCHS):
#             optimizer.zero_grad()

#             loss = residual_trajectory_matching(a)
#             loss.backward()
#             optimizer.step()

#             if epoch == EPOCHS - 1:
#                 last_residuals.append(loss.item())
#                 residual_trial.append(loss.item())

#     residual_samples.append(residual_trial)

# with open(OUTPUT_DIR / "residual_trajectory_matching.pkl", "wb") as f:
#     residual_obj = {
#         "num_frames_list": num_frames_list,
#         "residual_samples": residual_samples,
#     }
#     pkl.dump(residual_obj, f)


# """Derivative-Matching GD"""
# solution_derivative = torch.zeros_like(solution)
# solution_derivative[1:-1] = (solution[2:] - solution[:-2]) / (2 * solver.delta_t)
# solution_derivative[0] = (solution[1] - solution[0]) / solver.delta_t
# solution_derivative[-1] = (solution[-1] - solution[-2]) / solver.delta_t

# residual_samples = []

# for num_frames in tqdm(num_frames_list, desc="Varying # of Frames Trained Against", colour="green"):

#     residual_trial = []
#     for i in tqdm(range(TRIALS_PER_FRAME_COUNT), desc="Trials", colour="blue"):
#         frames = torch.linspace(0, solution.shape[0] - 1, num_frames).round().long()

#         def residual_derivative_matching(a):
#             x = solution[frames, :]
#             dx1 = a[0] * x[:, 0] - a[1] * x[:, 0] * x[:, 1]
#             dx2 = a[2] * x[:, 1] + a[3] * x[:, 0] * x[:, 1] - a[4] * x[:, 1]**2
#             pred_derivative = torch.stack([dx1, dx2], dim=1)
#             return torch.mean((pred_derivative - solution_derivative[frames, :])**2)

#         a = a_start_list[i].clone().detach().requires_grad_(True)
#         # a.requires_grad_(True)
#         optimizer = torch.optim.Adam([a], lr=1e-2)

#         for epoch in range(EPOCHS):
#             optimizer.zero_grad()

#             loss = residual_derivative_matching(a)
#             loss.backward()
#             optimizer.step()

#             if epoch == EPOCHS - 1:
#                 residual_trial.append(loss.item())

#     residual_samples.append(residual_trial)

# with open(OUTPUT_DIR / "residual_derivative_matching.pkl", "wb") as f:
#     residual_obj = {
#         "num_frames_list": num_frames_list,
#         "residual_samples": residual_samples,
#     }
#     pkl.dump(residual_obj, f)

# # x_frames = np.array(num_frames_list)

# # plt.figure(figsize=(10, 6))

# # plt.loglog(x_frames, last_residuals, marker="o", label="Trajectory-Matching GD", linewidth=5, markersize=8)
# # plt.loglog(x_frames, 1/(x_frames**2), "k--", label=r"Reference $x^{-2}$", linewidth=3)
# # plt.xlabel("# of True Frames Trained Against")
# # plt.ylabel(r"$\log$Residual")
# # plt.title("Convergence of Trajectory-Matching GD as # of True Frames Seen Increases")
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig(OUTPUT_DIR / "residual_trajectory_matching.svg")

"""Cascade Parameter Estimation with Population System"""

def model(u, a):
    x1 = u[:, 0]
    x2 = u[:, 1]

    dx1 = a[0] * x1 - a[1] * x1 * x2
    dx2 = a[2] * x2 + a[3] * x1 * x2 - a[4] * x2**2

    return torch.stack([dx1, dx2], dim=1)

def inverse_softplus(a, eps=1e-8):
    a = torch.clamp(a, min=eps)
    return torch.log(torch.expm1(a))

def cascade_loss(Phi, Phi_prime, C, raw_a, data, lambda_ode):
    u_hat = Phi @ C
    du_hat = Phi_prime @ C
    a = torch.nn.functional.softplus(raw_a)

    data_loss = torch.mean((u_hat - data)**2)
    ode_loss = torch.mean((du_hat - model(u_hat, a))**2)
    total_loss = data_loss + lambda_ode * ode_loss
    return total_loss, data_loss, ode_loss, a, u_hat

def rbf_basis(t_vals, num_basis, width_scale=1.0):
    t_min = t_vals[0]
    t_max = t_vals[-1]

    centers = torch.linspace(
        t_min,
        t_max,
        num_basis,
        dtype=t_vals.dtype,
        device=t_vals.device,
    )

    center_spacing = centers[1] - centers[0]
    width = width_scale * center_spacing

    t = t_vals[:, None]
    centers = centers[None, :]

    Phi = torch.exp(-0.5 * ((t - centers) / width) ** 2)
    Phi_dot = Phi * (-(t - centers) / width**2)

    return Phi, Phi_dot

def estimate_parameters_cascade(data, t_vals, a_start, lambda_ode, epochs, num_basis, lr):
    num_basis = min(num_basis, data.shape[0])
    Phi, Phi_prime = rbf_basis(t_vals, num_basis)

    C = torch.linalg.lstsq(Phi, data).solution
    C = C.clone().detach().requires_grad_(True)

    raw_a = inverse_softplus(a_start).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([C, raw_a], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss, data_loss, ode_loss, a, u_hat = cascade_loss(
            Phi,
            Phi_prime,
            C,
            raw_a,
            data,
            lambda_ode,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        loss, data_loss, ode_loss, a, u_hat = cascade_loss(
            Phi,
            Phi_prime,
            C,
            raw_a,
            data,
            lambda_ode,
        )

    return {
        "a_hat": a.detach(),
        "u_hat": u_hat.detach(),
        "total_loss": loss.item(),
        "data_loss": data_loss.item(),
        "ode_loss": ode_loss.item(),
    }

residual_samples = []
data_loss_samples = []
ode_loss_samples = []
trajectory_residual_samples = []
parameter_error_samples = []
a_hat_samples = []

for num_frames in tqdm(num_frames_list, desc="Cascade: varying # of frames", colour="green"):
    residual_trial = []
    data_loss_trial = []
    ode_loss_trial = []
    trajectory_residual_trial = []
    parameter_error_trial = []
    a_hat_trial = []

    frames = torch.linspace(0, solution.shape[0] - 1, num_frames).round().long()
    observed_solution = solution[frames, :]
    observed_t_vals = solver.t_vals[frames]

    for i in tqdm(range(TRIALS_PER_FRAME_COUNT), desc="Cascade trials", colour="blue"):
        result = estimate_parameters_cascade(
            observed_solution,
            observed_t_vals,
            a_start_list[i],
            lambda_ode=CASCADE_LAMBDA_ODE,
            epochs=EPOCHS,
            num_basis=CASCADE_NUM_BASIS,
            lr=CASCADE_LR,
        )

        a_hat = result["a_hat"]
        with torch.no_grad():
            estimated_solution = solver.solve(x0, a_hat)
            trajectory_residual = torch.mean((estimated_solution - solution)**2)
            parameter_error = torch.linalg.norm(a_hat - a_true) / torch.linalg.norm(a_true)

        residual_trial.append(result["total_loss"])
        data_loss_trial.append(result["data_loss"])
        ode_loss_trial.append(result["ode_loss"])
        trajectory_residual_trial.append(trajectory_residual.item())
        parameter_error_trial.append(parameter_error.item())
        a_hat_trial.append(a_hat.cpu())

    residual_samples.append(residual_trial)
    data_loss_samples.append(data_loss_trial)
    ode_loss_samples.append(ode_loss_trial)
    trajectory_residual_samples.append(trajectory_residual_trial)
    parameter_error_samples.append(parameter_error_trial)
    a_hat_samples.append(torch.stack(a_hat_trial))

with open(OUTPUT_DIR / "residual_cascade_parameter_estimation.pkl", "wb") as f:
    residual_obj = {
        "num_frames_list": list(num_frames_list),
        "residual_samples": residual_samples,
        "data_loss_samples": data_loss_samples,
        "ode_loss_samples": ode_loss_samples,
        "trajectory_residual_samples": trajectory_residual_samples,
        "parameter_error_samples": parameter_error_samples,
        "a_hat_samples": torch.stack(a_hat_samples),
        "a_start_list": torch.stack(a_start_list),
        "a_true": a_true,
        "x0": x0,
        "lambda_ode": CASCADE_LAMBDA_ODE,
        "epochs": EPOCHS,
        "num_basis": CASCADE_NUM_BASIS,
        "learning_rate": CASCADE_LR,
    }
    pkl.dump(residual_obj, f)
