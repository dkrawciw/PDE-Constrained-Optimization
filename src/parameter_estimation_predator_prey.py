import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from PopulationSystem import PopulationSystem

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
    "text.usetex": False,  # keep False to avoid requiring a LaTeX installation
    "mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRIALS_PER_FRAME_COUNT = 40
EPOCHS = 100
LEARNING_RATE = 1e-2
CASCADE_NUM_BASIS = 20
CASCADE_LAMBDA_ODE = 0.5
CASCADE_LR = 1e-2

dtype = torch.float64
solver = PopulationSystem(dtype=dtype)

def model(u, a):
    x1 = u[:, 0]
    x2 = u[:, 1]

    dx1 = a[0] * x1 - a[1] * x1 * x2
    dx2 = a[2] * x2 + a[3] * x1 * x2 - a[4] * x2**2

    return torch.stack([dx1, dx2], dim=1)


def inverse_softplus(a, eps=1e-8):
    a = torch.clamp(a, min=eps)
    return torch.log(torch.expm1(a))


def positive_parameter_perturbation(a, noise_scale):
    raw_a = inverse_softplus(a)
    return torch.nn.functional.softplus(raw_a + noise_scale * torch.randn_like(raw_a))


def finite_difference_derivative(solution, dt):
    solution_derivative = torch.zeros_like(solution)
    solution_derivative[1:-1] = (solution[2:] - solution[:-2]) / (2 * dt)
    solution_derivative[0] = (solution[1] - solution[0]) / dt
    solution_derivative[-1] = (solution[-1] - solution[-2]) / dt
    return solution_derivative


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

    if num_basis == 1:
        width = torch.clamp(t_max - t_min, min=torch.finfo(t_vals.dtype).eps)
    else:
        center_spacing = centers[1] - centers[0]
        width = width_scale * center_spacing

    t = t_vals[:, None]
    centers = centers[None, :]

    Phi = torch.exp(-0.5 * ((t - centers) / width) ** 2)
    Phi_prime = Phi * (-(t - centers) / width**2)

    return Phi, Phi_prime


def cascade_loss(Phi, Phi_prime, C, raw_a, data, lambda_ode):
    u_hat = Phi @ C
    du_hat = Phi_prime @ C
    a = torch.nn.functional.softplus(raw_a)

    data_loss = torch.mean((u_hat - data) ** 2)
    ode_loss = torch.mean((du_hat - model(u_hat, a)) ** 2)
    total_loss = data_loss + lambda_ode * ode_loss
    return total_loss, data_loss, ode_loss, a, u_hat


def fit_cascade(data, t_vals, a_start, epochs, num_basis, lambda_ode, lr):
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
        "total_loss": loss.item(),
        "data_loss": data_loss.item(),
        "ode_loss": ode_loss.item(),
    }


def fit_gradient_method(loss_fn, a_start, epochs, lr):
    raw_a = inverse_softplus(a_start).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw_a], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        a = torch.nn.functional.softplus(raw_a)
        loss = loss_fn(a)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        a = torch.nn.functional.softplus(raw_a)
        final_loss = loss_fn(a)

    return a.detach(), final_loss.item()


def selected_frame_indices(num_frames, total_frames):
    return torch.linspace(0, total_frames - 1, num_frames).round().long()


def evaluate_trajectory_residual(a_hat):
    with torch.no_grad():
        estimated_solution = solver.solve(initial_condition, a_hat)
        return torch.mean((estimated_solution - solution) ** 2).item()


def compute_residuals_trajectory(num_frames: int, epochs: int):
    frames = selected_frame_indices(num_frames, solution.shape[0])
    residuals = []

    for a_start in tqdm(a_start_list, desc="Running through Samples"):
        def loss_fn(a):
            estimated_solution = solver.solve(initial_condition, a)
            return torch.mean((estimated_solution[frames] - solution[frames]) ** 2)

        a_hat, training_loss = fit_gradient_method(
            loss_fn,
            a_start,
            epochs,
            LEARNING_RATE,
        )
        residuals.append(evaluate_trajectory_residual(a_hat))

    return residuals


def compute_residuals_derivative(num_frames: int, epochs: int):
    frames = selected_frame_indices(num_frames, solution.shape[0])
    observed_solution = solution[frames]
    observed_derivative = solution_derivative[frames]
    residuals = []

    for a_start in tqdm(a_start_list, desc="Running through Samples"):
        def loss_fn(a):
            predicted_derivative = model(observed_solution, a)
            return torch.mean((predicted_derivative - observed_derivative) ** 2)

        a_hat, training_loss = fit_gradient_method(
            loss_fn,
            a_start,
            epochs,
            LEARNING_RATE,
        )
        residuals.append(evaluate_trajectory_residual(a_hat))

    return residuals


def compute_residuals_cascade(num_frames: int, epochs: int):
    frames = selected_frame_indices(num_frames, solution.shape[0])
    observed_solution = solution[frames]
    observed_t_vals = solver.t_vals[frames]
    residuals = []

    for a_start in tqdm(a_start_list, desc="Running through Samples"):
        result = fit_cascade(
            observed_solution,
            observed_t_vals,
            a_start,
            epochs,
            CASCADE_NUM_BASIS,
            CASCADE_LAMBDA_ODE,
            CASCADE_LR,
        )
        residuals.append(evaluate_trajectory_residual(result["a_hat"]))

    return residuals


initial_condition = solver.default_initial_condition()
true_parameters = solver.default_parameters()

with torch.no_grad():
    solution = solver.solve(initial_condition, true_parameters)
    solution = solution + torch.randn_like(solution, dtype=dtype)       # Adding random noise to the 


solution_derivative = finite_difference_derivative(solution, solver.delta_t)

a_start_list = [
    positive_parameter_perturbation(true_parameters, noise_scale=0.5).detach()
    for _ in range(TRIALS_PER_FRAME_COUNT)
]

num_frames_list = range(5, solution.shape[0], 10)
epochs_list = range(5, EPOCHS, 5)

plotting_obj = {}

"""Trajectory-Matching"""
frame_samples_trajectory = []
for num_frames in tqdm(num_frames_list, desc="Trajectory frames"):
    current_sample = compute_residuals_trajectory(num_frames, EPOCHS)
    frame_samples_trajectory.append(current_sample)

epoch_samples_trajectory = []
for epochs in tqdm(epochs_list, desc="Trajectory epochs"):
    current_sample = compute_residuals_trajectory(solution.shape[0], epochs)
    epoch_samples_trajectory.append(current_sample)

trajectory_residual_obj = {
    "frame_samples_trajectory": frame_samples_trajectory,
    "epoch_samples_trajectory": epoch_samples_trajectory,
    "num_frames_list": list(num_frames_list),
    "epochs_list": list(epochs_list),
}

plotting_obj["trajectory_residual_obj"] = trajectory_residual_obj

"""Derivative-Matching"""
frame_samples_derivative = []
for num_frames in tqdm(num_frames_list, desc="Derivative frames"):
    current_sample = compute_residuals_derivative(num_frames, EPOCHS)
    frame_samples_derivative.append(current_sample)

epoch_samples_derivative = []
for epochs in tqdm(epochs_list, desc="Derivative epochs"):
    current_sample = compute_residuals_derivative(solution.shape[0], epochs)
    epoch_samples_derivative.append(current_sample)

derivative_residual_obj = {
    "frame_samples_trajectory": frame_samples_derivative,
    "epoch_samples_trajectory": epoch_samples_derivative,
    "num_frames_list": list(num_frames_list),
    "epochs_list": list(epochs_list),
}

plotting_obj["derivative_residual_obj"] = derivative_residual_obj

"""Cascade Parameter Estimation"""
frame_samples_cascade = []
for num_frames in tqdm(num_frames_list, desc="Cascade frames"):
    current_sample = compute_residuals_cascade(num_frames, EPOCHS)
    frame_samples_cascade.append(current_sample)

epoch_samples_cascade = []
for epochs in tqdm(epochs_list, desc="Cascade epochs"):
    current_sample = compute_residuals_cascade(solution.shape[0], epochs)
    epoch_samples_cascade.append(current_sample)

cascade_residual_obj = {
    "frame_samples_trajectory": frame_samples_cascade,
    "epoch_samples_trajectory": epoch_samples_cascade,
    "num_frames_list": list(num_frames_list),
    "epochs_list": list(epochs_list),
}

plotting_obj["cascade_residual_obj"] = cascade_residual_obj

plotting_obj["experiment_metadata"] = {
    "trials_per_frame_count": TRIALS_PER_FRAME_COUNT,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "cascade_num_basis": CASCADE_NUM_BASIS,
    "cascade_lambda_ode": CASCADE_LAMBDA_ODE,
    "cascade_learning_rate": CASCADE_LR,
    "initial_condition": initial_condition.cpu(),
    "true_parameters": true_parameters.cpu(),
    "a_start_list": torch.stack(a_start_list).cpu(),
}

output_path = OUTPUT_DIR / "parameter_estimation_predator_prey_with_noise.pkl"
temp_output_path = output_path.with_suffix(".tmp")
with open(temp_output_path, "wb") as pkl_file:
    pkl.dump(plotting_obj, pkl_file)
temp_output_path.replace(output_path)

print("File Successfully Saved!")
