import pickle as pkl
from pathlib import Path

import torch
from tqdm import tqdm

from PopulationSystem import PopulationSystem

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRIALS_PER_EPOCH_COUNT = 40
EPOCHS_LIST = list(range(10, 210, 10))
LEARNING_RATE = 1e-2
CASCADE_NUM_BASIS = 20
CASCADE_LAMBDA_ODE = 0.5
CASCADE_LR = 1e-2

dtype = torch.float64


def model(u, a):
    x1 = u[:, 0]
    x2 = u[:, 1]

    dx1 = a[0] * x1 - a[1] * x1 * x2
    dx2 = a[2] * x2 + a[3] * x1 * x2 - a[4] * x2**2

    return torch.stack([dx1, dx2], dim=1)


def inverse_softplus(a, eps=1e-8):
    a = torch.clamp(a, min=eps)
    return torch.log(torch.expm1(a))


def finite_difference_derivative(solution, dt):
    solution_derivative = torch.zeros_like(solution)
    solution_derivative[1:-1] = (solution[2:] - solution[:-2]) / (2 * dt)
    solution_derivative[0] = (solution[1] - solution[0]) / dt
    solution_derivative[-1] = (solution[-1] - solution[-2]) / dt
    return solution_derivative


def trajectory_matching_loss(solver, x0, solution, a):
    predicted_solution = solver.solve(x0, a)
    return torch.mean((predicted_solution - solution) ** 2)


def derivative_matching_loss(solution, solution_derivative, a):
    predicted_derivative = model(solution, a)
    return torch.mean((predicted_derivative - solution_derivative) ** 2)


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


def fit_gradient_method(loss_fn, a_start, epochs, lr):
    a = a_start.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([a], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(a)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(a)

    return a.detach(), final_loss.item()


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


def evaluate_estimate(solver, x0, solution, a_hat, a_true):
    with torch.no_grad():
        estimated_solution = solver.solve(x0, a_hat)
        trajectory_residual = torch.mean((estimated_solution - solution) ** 2)
        parameter_error = torch.linalg.norm(a_hat - a_true) / torch.linalg.norm(a_true)

    return trajectory_residual.item(), parameter_error.item()


def append_method_result(results, method_name, epoch_index, trial_result):
    for key, value in trial_result.items():
        results[method_name][key][epoch_index].append(value)


solver = PopulationSystem()
x0 = torch.rand(2, dtype=dtype)
a_true = torch.nn.functional.softplus(torch.randn(5, dtype=dtype))

with torch.no_grad():
    solution = solver.solve(x0, a_true)

solution_derivative = finite_difference_derivative(solution, solver.delta_t)
a_start_list = [
    a_true + 0.5 * torch.randn(5, dtype=dtype)
    for _ in range(TRIALS_PER_EPOCH_COUNT)
]

methods = ["trajectory_matching", "derivative_matching", "cascade"]
results = {
    method: {
        "training_loss": [[] for _ in EPOCHS_LIST],
        "trajectory_residual": [[] for _ in EPOCHS_LIST],
        "parameter_error": [[] for _ in EPOCHS_LIST],
        "a_hat": [[] for _ in EPOCHS_LIST],
    }
    for method in methods
}
results["cascade"]["data_loss"] = [[] for _ in EPOCHS_LIST]
results["cascade"]["ode_loss"] = [[] for _ in EPOCHS_LIST]

for epoch_index, epochs in enumerate(tqdm(EPOCHS_LIST, desc="Varying training epochs", colour="green")):
    for trial_index in tqdm(range(TRIALS_PER_EPOCH_COUNT), desc="Trials", colour="blue"):
        a_start = a_start_list[trial_index]

        trajectory_loss_fn = lambda a: trajectory_matching_loss(solver, x0, solution, a)
        a_hat, training_loss = fit_gradient_method(
            trajectory_loss_fn,
            a_start,
            epochs,
            LEARNING_RATE,
        )
        trajectory_residual, parameter_error = evaluate_estimate(solver, x0, solution, a_hat, a_true)
        append_method_result(
            results,
            "trajectory_matching",
            epoch_index,
            {
                "training_loss": training_loss,
                "trajectory_residual": trajectory_residual,
                "parameter_error": parameter_error,
                "a_hat": a_hat.cpu(),
            },
        )

        derivative_loss_fn = lambda a: derivative_matching_loss(solution, solution_derivative, a)
        a_hat, training_loss = fit_gradient_method(
            derivative_loss_fn,
            a_start,
            epochs,
            LEARNING_RATE,
        )
        trajectory_residual, parameter_error = evaluate_estimate(solver, x0, solution, a_hat, a_true)
        append_method_result(
            results,
            "derivative_matching",
            epoch_index,
            {
                "training_loss": training_loss,
                "trajectory_residual": trajectory_residual,
                "parameter_error": parameter_error,
                "a_hat": a_hat.cpu(),
            },
        )

        cascade_result = fit_cascade(
            solution,
            solver.t_vals,
            a_start,
            epochs,
            CASCADE_NUM_BASIS,
            CASCADE_LAMBDA_ODE,
            CASCADE_LR,
        )
        a_hat = cascade_result["a_hat"]
        trajectory_residual, parameter_error = evaluate_estimate(solver, x0, solution, a_hat, a_true)
        append_method_result(
            results,
            "cascade",
            epoch_index,
            {
                "training_loss": cascade_result["total_loss"],
                "trajectory_residual": trajectory_residual,
                "parameter_error": parameter_error,
                "a_hat": a_hat.cpu(),
                "data_loss": cascade_result["data_loss"],
                "ode_loss": cascade_result["ode_loss"],
            },
        )

for method in methods:
    results[method]["a_hat"] = torch.stack([
        torch.stack(epoch_a_hats)
        for epoch_a_hats in results[method]["a_hat"]
    ])

with open(OUTPUT_DIR / "population_parameter_estimation_by_epochs.pkl", "wb") as f:
    pkl.dump(
        {
            "epochs_list": EPOCHS_LIST,
            "trials_per_epoch_count": TRIALS_PER_EPOCH_COUNT,
            "results": results,
            "a_start_list": torch.stack(a_start_list),
            "a_true": a_true,
            "x0": x0,
            "solution": solution,
            "learning_rate": LEARNING_RATE,
            "cascade_num_basis": CASCADE_NUM_BASIS,
            "cascade_lambda_ode": CASCADE_LAMBDA_ODE,
            "cascade_learning_rate": CASCADE_LR,
        },
        f,
    )
