# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Traulls is a Julia solver for **bound- and equality-constrained nonlinear least-squares**:

```
minₓ ½ r(x)ᵀr(x)   s.t.  c(x) = 0,  Ax = b,  ℓ ≤ x ≤ u
```

It uses an **outer augmented Lagrangian** loop whose subproblems are solved by an **inner trust-region gradient-projection** method. The single exported entry point is `traulls(model; kwargs...)`.

## Commands

This is a standard Julia package (the working directory is the dev'd package under `~/.julia/dev`).

```bash
# Run the full test suite
julia --project -e 'using Pkg; Pkg.test()'

# Start a REPL with the package activated (fastest for iterating)
julia --project
```

To run a **single test file** interactively, activate the project and include it from a REPL where `Traulls` is loaded:

```julia
using Traulls, LinearAlgebra, ForwardDiff, SparseArrays, Test
include("test/solve_hs65.jl")
```

`test/runtests.jl` selects which test files run — entries are toggled by commenting/uncommenting `include(...)` lines. **Check it before assuming a test file is wired in**; most files there are currently commented out and only `sparse_model.jl` runs by default. When adding tests, uncomment or add the corresponding `include`.

## Architecture

`src/Traulls.jl` is the module root and the `include` order is load-bearing (types and helpers must be defined before files that use them). Three abstract types defined there anchor the dispatch design:

- `AbstractCnlsModel{T}` — the problem model
- `ALHessian{T}` — augmented-Lagrangian Hessian approximations
- `Projector{T}` — projection onto feasible subspaces

### Solver flow (`solver.jl`)

1. **`traulls`** — outer AL loop. Each outer iteration fixes the penalty `μ` and multipliers `y`, solves the AL subproblem, then either updates multipliers (`first_order_multipliers!`) and tightens tolerances if feasibility improved, or increases `μ` otherwise. Terminates with a `CriticalityStatus`.
2. **`solve_subproblem!`** — inner trust-region minimization of the AL for fixed `(y, μ)`. Builds a quadratic model `qₖ(s) = ½sᵀHₖs + sᵀgₖ`, takes a step, accepts/rejects by the actual-vs-predicted reduction ratio, updates the trust-region radius (`trust_region.jl`) and the Hessian approximation.
3. **`projected_gradient!`** — computes the step: first a **Cauchy point** along the projected-gradient path (`cauchy.jl`), then refines within the free subspace via **projected conjugate gradient** (`cg.jl`, `pcg!`).

### Models (`model.jl`, `sparse_model.jl`)

`CnlsModel{T}` (dense) and `SparseCnlsModel{T,Ti}` (sparse) both subtype `AbstractCnlsModel`. They store user-supplied **in-place** evaluation closures (`res!`, `nleq!`, `nlineq!`, and their Jacobians) plus dimensions, bounds, linear constraints, a starting point, and a `TraullsCounters`.

Key conventions:

- **In-place vs out-of-place constructors**: `CnlsModel!(...)` takes in-place functions with signature `f!(fx, x)` returning `nothing`; `CnlsModel(...)` takes out-of-place `f(x)` returning a vector and wraps them. Same pattern for `SparseCnlsModel!`.
- **Inequalities become equalities via slack variables**: `g(x) ≥ 0` is rewritten as `g(x) − u = 0, u ≥ 0`. Slack vars are appended to `x`, so `n = nvar + nslack`; bounds and the linear-constraint matrix `A` are padded accordingly. This happens inside the constructors — downstream code only ever sees equality-constrained problems.
- **Constraint-type tag**: constructors for single-type problems take `Val(:only_equalities)` or `Val(:only_inequalities)` (the `ConstraintsType` union) to dispatch the slack-handling.
- **`SparseCnlsModel` requires explicit sparsity patterns** (`*_nzrows`/`*_nzcols`); the user's Jacobian functions *must* write only into that pattern. Sparse and dense models share the same generic `residuals!`/`jac_residuals!`/etc. interface, so the solver is agnostic to which is passed.

### Hessian approximations (`hessian.jl` + `sr1.jl`, `bfgs.jl`, `lsr1.jl`)

The `HessianApprox` enum (`gn`, `sr1`, `bfgs`, `hybrid_bfgs`, `hybrid_sr1`, `limited_sr1`) and the `dict_hessians` symbol→enum map drive selection from the `hessian_approx` keyword. Each variant has a struct subtyping `ALHessian` (e.g. `GN`, `SR1`, `BFGS`, `HybridBFGS`, `HybridSR1`, `LSR1`).

These operators **never form the Hessian explicitly** — they overload `LinearAlgebra.mul!` and `Base.:*` to compute Hessian-vector products lazily (e.g. Gauss-Newton's `H = JᵀJ + μCᵀC` is applied as two mat-vec passes through `J` and `C`). When adding/modifying a variant, implement `mul!`, `update_hessian!`, and `reset_hessian!`, and wire it into the `@match` blocks in `solver.jl` (both the construction in `traulls` and the reset/update in `solve_subproblem!`) plus `dict_hessians`. The `update_jacobians!` helper has separate methods for dense matrices and `SparseMatrixCSC` (the latter copies `.nzval` to preserve the sparsity pattern).

### Projectors (`polyhedral_constraints.jl`)

Two `Projector` implementations handle the two linear-constraint regimes:

- **`CoordinateSubspaceProjector`** — used when linear constraints are *only bounds*. Projection just zeroes fixed components; cheap.
- **`SubspaceProjector`** — used when general linear equalities `Ax = b` are present. Projects onto `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}` by solving the normal equations `A₊A₊ᵀ y = A₊x` through a Cholesky factor `L` of the augmented Gram matrix. That factor is maintained **incrementally**: `set_active!` appends a row/column via a bordered Cholesky update (`append_active!`, O((m+p)²)) rather than refactorizing; `L` lives in a single preallocated `n×n` buffer with a current-dimension field `naug`, so `reset_projector!` is O(1) and `mul!` is allocation-free. `set_free!` (a downdate) rebuilds the factor for the remaining active set, and `identify_active_set!` rebuilds from a reset. The struct is parametrized on the concrete equality-matrix type to avoid dynamic dispatch in the projection hot path.

The solver picks the projector in `initial_point_and_projector!` based on `model.nlincons > 0`, and branches on this `lincons_present` flag throughout to choose the matching `criticality_measure` method. When `A` is present, the initial point is first made linear-feasible by solving an L1 feasibility LP with HiGHS/JuMP (`solve_linfeas_pb!`).

### Supporting files

- `al_utils.jl` — augmented-Lagrangian objective/gradient helpers (`al_obj`, `al_grad`, `al_objgrad!`, `least_squares_multipliers`, `first_order_multipliers!`).
- `workspace.jl` — `Workspace{T}`, a bag of preallocated buffer vectors threaded through the inner loop to avoid reallocation. Add a field here rather than allocating inside hot loops.
- `execution_metrics.jl` — `TraullsCounters` (evaluation/iteration counts), `TraullsResults{T}` (returned to the caller), and the `CriticalityStatus` enum.
- `trust_region.jl` — `TrustRegion{T}` and the radius-update / step-acceptance logic.
- `print_info.jl` — verbose iteration logging (gated by `verbose` / `inner_verbose`).

## Conventions

- Everything is **parametrized on the element type `T`** (and `Ti` for sparse index types). Preserve this — write `T(10)` not `10.0`, allocate with `zeros(T, ...)`, etc.
- The codebase favors **in-place mutation** with preallocated buffers; `!`-suffixed functions mutate their first argument(s) and typically return `nothing` or a scalar measure, not the mutated array.
- `Match.@match` is used for enum dispatch on Hessian type. `LinearAlgebra.mul!`, `Base.:*`, and `transpose` are overloaded for the custom operator/matrix types — prefer extending these over writing bespoke product functions.
- The `draft/` directory holds the LaTeX paper describing the method; it is not part of the package and is irrelevant to the build/tests.
