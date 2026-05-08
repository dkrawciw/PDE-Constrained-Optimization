# PDE Constrained Optimization

**School:** Colorado School of Mines

**Class:** Numerical Optimization

**Professor:** Dr. Samy Wu Fung

**Author:** Daniel Krawciw

## Project Description

I have take scientific computing classes and I have taken numerical optimization classes. I wanted to learn about how to combine the two fields and learn about constrained optimization on ODEs and PDEs. This project represents my research and first attempts with different methods of estimating parameters from data for mathematical models.

## Important scripts/notebooks

Really the only significant files to work with are:

1. `src/parameter_estimation_predator_prey.py`
2. `src/parameter_estimation_advection_diffusion.py`
3. `notebooks/parameter_estimation.ipynb`

Using these 3 scripts/notebooks, one can replicate my paper results. The `.tex` file for my final report can be found in `paper/main.tex`.

## Instructions

Ensure that you have [Astral UV](https://docs.astral.sh/uv/#installation) installed before beginning.

Start by running the following commands:

```bash
uv sync
source .venv/bin/activate
```

Then generate the data using:

```bash
uv run python src/parameter_estimation_predator_prey.py
uv run python src/parameter_estimation_advection_diffusion.py
```

After this, you can explore the data and reproduce my plots in the notebook: `notebooks/parameter_estimation.ipynb`.