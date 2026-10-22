# Numerical experiments of the paper

The numerical results of the paper *A structured quasi-Newton augmented Lagrangian algorithm
for nonlinear least squares with linear and nonlinear constraints* (P. Borie, F. Bastin and
S. Dellacherie) come from the version of this repository tagged `coap-2026`. Everything they
involve is in the `benchmark` folder, and the manuscript is in `preprint`.

## Environment

The files `benchmark/Project.toml` and `benchmark/Manifest.toml` pin every package to the
version used for the experiments, with Julia 1.12.1; TRAULLS itself is taken from the
repository. From the `benchmark` folder,

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

installs this environment. All the commands below are run from the same folder.

## Result files

The folder `benchmark/results` holds the result files of all the runs reported in the paper,
so that the figures, the tables and the numbers of the text can be rebuilt without running
the solvers again. Each script that runs a solver records the commit of the solver code it
used, and refuses to run if `src` contains uncommitted changes.

| Files | Content |
|:------|:--------|
| `traulls_gn.csv`, `traulls_sr1.csv`, `traulls_bfgs.csv`, `traulls_hybrid_sr1.csv`, `traulls_hybrid_bfgs.csv` | TRAULLS with each Hessian approximation |
| `ipopt.csv`, `ipopt-lbfgs.csv`, `percival.csv` | Ipopt with exact second derivatives, Ipopt with L-BFGS, Percival |
| `commit.txt` | commit of the solver code of the files above |
| `linear_constraints.csv`, `linear_constraints_commit.txt` | linear constraints kept explicit or penalized (Section 4.3) |
| `secant_comparison.csv`, `secant_comparison_mu.csv`, `secant_comparison_commit.txt` | structured and standard secant equations (Section 2.4) |

## Scripts

| Script | What it does | Duration |
|:-------|:-------------|:---------|
| `run_experiments.jl` | Timed campaign of Sections 4.1 and 4.2: every solver on the 82 instances, each solved once to warm up, then three times, keeping the smallest time. It includes `run_traulls.jl` and `run_percival_ipopt.jl`, and writes its files to `new_results`. | about 1 h 40 |
| `make_figures.jl` | Performance profiles of Figures 2, 3, 4 and 6, saved in `preprint/figures`. They can also be explored in the notebook `profiles.ipynb`. | a few minutes |
| `residual_cost_analysis.jl` | Table 1, break-even costs and Figure 5 of Section 4.1. | a few minutes |
| `paper_numbers.jl` | The other numbers quoted in Sections 2.4, 4.2 and 4.3: ratios against Ipopt and Percival, comparison of the two formulations of the linear constraints and of the two secant equations. | a few minutes |
| `linear_constraints/run_linear_constraints.jl` | The 15 problems of Section 4.3 under both formulations (models in `linear_constraints/models.jl`), with every Hessian approximation; Table 2 is the Hybrid-SR1 part of its output. | about a minute |
| `secant_comparison/run_secant_comparison.jl` | The 82 instances with the four dense secant approximations, under the structured and the standard secant equations (Section 2.4). | about 2 h 20 |

The durations were measured on the machine of the paper, an Intel Core Ultra 9 285 processor.
Most of the time of `run_experiments.jl` is spent by Percival on the two instances where it
reaches its time limit of 10 minutes, and most of that of `run_secant_comparison.jl` by the
plain SR1 update with the standard secant equation, which fails on the instances of problem
5.15 with ``n = 500`` and ``n = 1000``.

The figure files of `preprint/figures` correspond to the figures of the paper as follows.

| Figure of the paper | Files |
|:--------------------|:------|
| 2 | `fig1_*`, `fig3_*`, `fig2_*` |
| 3 | `fig8_*` to `fig11_*` |
| 4 | `fig4_*` to `fig7_*` |
| 5 | `fig15_*` |
| 6 | `fig13_*`, `fig14_*`, `fig12_*` |

## Other files

- `traulls_models`: the input models for TRAULLS.
- `corrected_hs65_nls_problems.jl`: problem HS65 with its inequality constraint, as in the
  model of TRAULLS, for Ipopt and Percival.
- `nlpmodelsjump_fix.jl`: fix of the Hessian of the constrained least-squares models in
  NLPModelsJuMP 0.13.6, without which Ipopt with exact second derivatives fails on them.

The iteration and evaluation counts do not depend on the machine. The times do, and they also
vary between two runs on the same machine, by up to about 30 % on the largest instances with
the dense secant approximations.

## Rebuilding the figures and the numbers

Without running any solver:

```
julia --project=. make_figures.jl
julia --project=. residual_cost_analysis.jl
julia --project=. paper_numbers.jl
```
