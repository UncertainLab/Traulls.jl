# Traulls

[![Build Status](https://github.com/UncertainLab/Traulls.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UncertainLab/Traulls.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UncertainLab/Traulls.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/UncertainLab/Traulls.jl)

`Traulls.jl` is a Julia solver for constrained nonlinear least-squares problems. It uses an
Augmented Lagrangian outer loop combined with an inner trust-region gradient-projection
method.

A theoritical description of the method, together with a convergence analysis and numerical experiments, are given in the following [preprint](https://arxiv.org/abs/2607.11239) available on arXiv.

## Installation

The package can be downloaded directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/UncertainLab/Traulls.jl")
```

## Problem formulation

Traulls solves problems of the form

```
minₓ   ½·r(x)ᵀr(x)
s.t.   h(x) = 0
       g(x) ≥ 0
       A·x = b
       ℓ ≤ x ≤ u
```

where `r` is the vector of residuals, `h` and `g` are the nonlinear equality and inequality
constraints (twice continuously differentiable), `A·x = b` are linear equality constraints,
and `ℓ, u` are bounds on the variables.

A problem is described by a `CnlsModel`, built from functions evaluating the residuals and
constraints together with their Jacobians. Both out-of-place (`CnlsModel`, functions of the
form `f(x)`) and in-place (`CnlsModel!`, functions of the form `f!(fx, x)`) variants are
available. The model is passed to `traulls`, which returns a `TraullsResults` holding the
solution, the Lagrange multipliers, the final objective, feasibility and criticality
measures, the termination status, and execution counters.

## Example

The following builds and solves problem 65 from the Hock & Schittkowski collection:

```julia
using Traulls

# Dimensions
n = 3   # variables
m = 3   # residuals
p = 1   # constraints

# Residuals and their Jacobian
r(x)     = [x[1] - x[2],
            (x[1] + x[2] - 10) / 3,
            x[3] - 5.0]
jac_r(x) = [1.0  -1.0  0.0;
            1/3   1/3  0.0;
            0.0   0.0  1.0]

# Constraint and its Jacobian
c(x)     = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
jac_c(x) = [-2x[1] -2x[2] -2x[3]]

# Bounds and starting point
x_low = [-4.5, -4.5, -5.0]
x_upp = [ 4.5,  4.5,  5.0]
x0    = [-5.0,  5.0,  0.0]

model = CnlsModel(r, c, jac_r, jac_c, x_low, x_upp, x0, n, m, p, Val(:only_inequalities))

results = traulls(model; verbose=true)
```

## Benchmarks and further examples

Benchmarks comparing Traulls to other solvers can be found in the `benchmark` folder. Other
examples of model implementations are provided in the `benchmark/traulls_models` folder.




