import numpy as np
import torch

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
EPOCHS = 100
LEARNING_RATE = 5e-2
DIFFUSION_STABILITY_SAFETY = 0.95

dtype = torch.float64
solver = PeriodicAdvectionDiffusion()
MAX_DIFFUSION_COEFFICIENT = DIFFUSION_STABILITY_SAFETY * solver.h**2 / (4 * solver.delta_t)


def bounded_diffusion(raw_D):
    return MAX_DIFFUSION_COEFFICIENT * torch.sigmoid(raw_D)


def inverse_bounded_diffusion(D, eps=1e-8):
    scaled_D = torch.clamp(D / MAX_DIFFUSION_COEFFICIENT, min=eps, max=1 - eps)
    return torch.logit(scaled_D)


def perturb_diffusion_coefficient(D, noise_scale):
    raw_D = inverse_bounded_diffusion(D)
    return bounded_diffusion(raw_D + noise_scale * torch.randn_like(raw_D))

initial_condition = solver.gaussian_initial_condition()
diffusion_coefficient = torch.tensor(
    0.25,
    dtype=solver.dtype,
    device=solver.device,
    requires_grad=True,
)
with torch.no_grad():
    solution = solver.solve(initial_condition, diffusion_coefficient, show_progress=False)

D_start_list = []
for _ in range(TRIALS_PER_FRAME_COUNT):
    D_start = perturb_diffusion_coefficient(diffusion_coefficient, noise_scale=1.0)
    D_start_list.append(D_start.detach())

num_frames_list = range(5, solution.shape[0], 10)
epochs_list = range(5, EPOCHS, 5)
t_vals = solver.t_range[0] + solver.delta_t * torch.arange(
    solution.shape[0],
    dtype=solver.dtype,
    device=solver.device,
)

plotting_obj = {}

"""Trajectory-Matching"""
def compute_residuals_trajectory(num_frames: int, epochs: int):
    residuals = []
    for D_start in tqdm(D_start_list, desc="Running through Samples"):
        raw_D = inverse_bounded_diffusion(D_start).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw_D], lr=LEARNING_RATE)

        for epoch in range(epochs):
            optimizer.zero_grad()
            D_estimate = bounded_diffusion(raw_D)
            estimated_solution = solver.solve(initial_condition, D_estimate, show_progress=False)
            loss = torch.mean((estimated_solution[:num_frames] - solution[:num_frames]) ** 2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            D_estimate = bounded_diffusion(raw_D)
            final_estimated_solution = solver.solve(initial_condition, D_estimate, show_progress=False)
            final_residual = torch.mean((final_estimated_solution - solution) ** 2).item()
            residuals.append(final_residual)

    return residuals

frame_samples_trajectory = []
for num_frames in tqdm(num_frames_list, desc="Frames"):
    current_sample = compute_residuals_trajectory(num_frames, EPOCHS)
    frame_samples_trajectory.append(current_sample)

epoch_samples_trajectory = []
for epochs in tqdm(epochs_list, desc="Epochs"):
    current_sample = compute_residuals_trajectory(num_frames_list[-1], epochs)
    epoch_samples_trajectory.append(current_sample)

trajectory_residual_obj = {
    "frame_samples_trajectory": frame_samples_trajectory,
    "epoch_samples_trajectory": epoch_samples_trajectory,
    "num_frames_list": list(num_frames_list),
    "epochs_list": list(epochs_list),
}

plotting_obj["trajectory_residual_obj"] = trajectory_residual_obj

"""Derivative-Matching"""
solution_derivative = torch.zeros_like(solution)
solution_derivative[1:-1] = (solution[2:] - solution[:-2]) / (2 * solver.delta_t)
solution_derivative[0] = (solution[1] - solution[0]) / solver.delta_t
solution_derivative[-1] = (solution[-1] - solution[-2]) / solver.delta_t

def compute_residuals_derivative(num_frames: int, epochs: int):
    residuals = []
    for D_start in D_start_list:
        raw_D = inverse_bounded_diffusion(D_start).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw_D], lr=LEARNING_RATE)

        for epoch in range(epochs):
            optimizer.zero_grad()
            D_estimate = bounded_diffusion(raw_D)

            predicted_derivative = torch.stack([
                solver.rhs(
                    u=solution[i],
                    t=t_vals[i],
                    diffusion_coefficient=D_estimate,
                )
                for i in range(num_frames)
            ])
            loss = torch.mean(
                (predicted_derivative - solution_derivative[:num_frames]) ** 2
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            D_estimate = bounded_diffusion(raw_D)
            final_estimated_solution = solver.solve(initial_condition, D_estimate, show_progress=False)
            final_residual = torch.mean((final_estimated_solution - solution) ** 2).item()
            residuals.append(final_residual)

    return residuals

frame_samples_trajectory = []
for num_frames in tqdm(num_frames_list, desc="Frames"):
    current_sample = compute_residuals_derivative(num_frames, EPOCHS)
    frame_samples_trajectory.append(current_sample)

epoch_samples_trajectory = []
for epochs in tqdm(epochs_list, desc="Epochs"):
    current_sample = compute_residuals_derivative(num_frames_list[-1], epochs)
    epoch_samples_trajectory.append(current_sample)

derivative_residual_obj = {
    "frame_samples_trajectory": frame_samples_trajectory,
    "epoch_samples_trajectory": epoch_samples_trajectory,
    "num_frames_list": list(num_frames_list),
    "epochs_list": list(epochs_list),
}

plotting_obj["derivative_residual_obj"] = derivative_residual_obj

output_path = OUTPUT_DIR / "parameter_estimation_advection_diffusion.pkl"
temp_output_path = output_path.with_suffix(".tmp")
with open(temp_output_path, "wb") as pkl_file:
    pkl.dump(plotting_obj, pkl_file)
temp_output_path.replace(output_path)

print("File Successfully Saved!")
