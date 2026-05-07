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

# a_start = a_true + torch.randn(5, dtype=dtype)
a_start_list = []
for _ in range(TRIALS_PER_FRAME_COUNT):
    a_start = a_true + torch.randn(5, dtype=dtype)*0.5
    a_start_list.append(a_start)

last_residuals = []
num_frames_list = range(5, solution.shape[0], 10)

residual_samples = []

for num_frames in tqdm(num_frames_list, desc="Varying # of Frames Trained Against", colour="green"):

    residual_trial = []
    for i in tqdm(range(TRIALS_PER_FRAME_COUNT), desc="Trials", colour="blue"):
        frames = torch.linspace(0, solution.shape[0] - 1, num_frames).round().long()
        residual_trajectory_matching = lambda a: torch.mean((solver.solve(x0, a)[frames,:] - solution[frames,:])**2)

        a = a_start_list[i].clone().detach().requires_grad_(True)
        # a.requires_grad_(True)
        optimizer = torch.optim.Adam([a], lr=1e-2)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            loss = residual_trajectory_matching(a)
            loss.backward()
            optimizer.step()

            if epoch == EPOCHS - 1:
                last_residuals.append(loss.item())
                residual_trial.append(loss.item())

    residual_samples.append(residual_trial)

with open(OUTPUT_DIR / "residual_trajectory_matching.pkl", "wb") as f:
    residual_obj = {
        "num_frames_list": num_frames_list,
        "residual_samples": residual_samples,
    }
    pkl.dump(residual_obj, f)


"""Derivative-Matching GD"""
solution_derivative = torch.zeros_like(solution)
solution_derivative[1:-1] = (solution[2:] - solution[:-2]) / (2 * solver.delta_t)
solution_derivative[0] = (solution[1] - solution[0]) / solver.delta_t
solution_derivative[-1] = (solution[-1] - solution[-2]) / solver.delta_t

residual_samples = []

for num_frames in tqdm(num_frames_list, desc="Varying # of Frames Trained Against", colour="green"):

    residual_trial = []
    for i in tqdm(range(TRIALS_PER_FRAME_COUNT), desc="Trials", colour="blue"):
        frames = torch.linspace(0, solution.shape[0] - 1, num_frames).round().long()

        def residual_derivative_matching(a):
            x = solution[frames, :]
            dx1 = a[0] * x[:, 0] - a[1] * x[:, 0] * x[:, 1]
            dx2 = a[2] * x[:, 1] + a[3] * x[:, 0] * x[:, 1] - a[4] * x[:, 1]**2
            pred_derivative = torch.stack([dx1, dx2], dim=1)
            return torch.mean((pred_derivative - solution_derivative[frames, :])**2)

        a = a_start_list[i].clone().detach().requires_grad_(True)
        # a.requires_grad_(True)
        optimizer = torch.optim.Adam([a], lr=1e-2)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            loss = residual_derivative_matching(a)
            loss.backward()
            optimizer.step()

            if epoch == EPOCHS - 1:
                residual_trial.append(loss.item())

    residual_samples.append(residual_trial)

with open(OUTPUT_DIR / "residual_derivative_matching.pkl", "wb") as f:
    residual_obj = {
        "num_frames_list": num_frames_list,
        "residual_samples": residual_samples,
    }
    pkl.dump(residual_obj, f)

# x_frames = np.array(num_frames_list)

# plt.figure(figsize=(10, 6))

# plt.loglog(x_frames, last_residuals, marker="o", label="Trajectory-Matching GD", linewidth=5, markersize=8)
# plt.loglog(x_frames, 1/(x_frames**2), "k--", label=r"Reference $x^{-2}$", linewidth=3)
# plt.xlabel("# of True Frames Trained Against")
# plt.ylabel(r"$\log$Residual")
# plt.title("Convergence of Trajectory-Matching GD as # of True Frames Seen Increases")
# plt.legend()
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "residual_trajectory_matching.svg")
