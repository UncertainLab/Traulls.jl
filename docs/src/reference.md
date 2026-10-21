# [Reference](@id Reference)

```@meta
CurrentModule = Traulls
```

## Index

```@index
Pages = ["reference.md"]
```

## Problem models

Models of constrained nonlinear least-squares problems, with dense ([`CnlsModel`](@ref)) or
sparse ([`SparseCnlsModel`](@ref)) Jacobians, and the functions evaluating their residuals,
constraints and Jacobians.

```@autodocs
Modules = [Traulls]
Pages = ["model.jl", "sparse_model.jl"]
```

## Optimization levels

The algorithm is made of nested levels, from the
[augmented Lagrangian outer loop](@ref method_al) down to the
[conjugate gradient iterations](@ref method_qp).

### Augmented Lagrangian outer loop

```@docs
traulls
```

### Trust-region inner loop

See [Solving the subproblem of an outer iteration](@ref method_subproblem).

```@docs
solve_subproblem!
```

### Quadratic program

See [Solving the quadratic program](@ref method_qp).

```@docs
projected_gradient!
```

### Cauchy step

```@docs
cauchy_step!
initial_fixed!
initial_fixed
next_breakpoint
```

### Projected conjugate gradient

```@docs
pcg!
CG_status
```

## Criticality measure

See [Stopping criteria](@ref method_stop).

```@docs
criticality_measure
```

## Projections

Projections onto the null space of the linear constraints and of the active bounds,
and computation of a feasible [initial point](@ref method_init).

```@autodocs
Modules = [Traulls]
Pages = ["polyhedral_constraints.jl"]
```

### Feasible initial point

```@docs
initial_point_and_projector!
solve_linfeas_pb!
```

## Hessian approximations

See [Hessian approximation](@ref method_hessian).

```@autodocs
Modules = [Traulls]
Pages = ["hessian.jl", "sr1.jl", "bfgs.jl", "lsr1.jl"]
```

## Trust region

```@autodocs
Modules = [Traulls]
Pages = ["trust_region.jl"]
```

## Augmented Lagrangian utilities

Objective and gradient of the augmented Lagrangian, multiplier estimates and tolerances.

```@autodocs
Modules = [Traulls]
Pages = ["al_utils.jl"]
```

## Results and workspace

```@autodocs
Modules = [Traulls]
Pages = ["execution_metrics.jl", "workspace.jl"]
```
