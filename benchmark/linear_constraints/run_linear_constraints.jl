# Linear constraints kept explicit (formulation E) or handled by the augmented Lagrangian
# (formulation P), on the problems of models.jl (Section 4.3 of the paper). Iteration counts
# do not depend on timings, so that each problem is solved once, with every Hessian
# approximation, the tolerances of run_traulls.jl, and ε-active sets on (default) and off for
# formulation E.
#
# Usage, from the benchmark folder:
#     julia --project=. linear_constraints/run_linear_constraints.jl

include(joinpath(@__DIR__, "..", "..", "src", "Traulls.jl"))
using .Traulls
using LinearAlgebra, Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OPT_CRIT = 1e-5
const FEAS_CRIT = 1e-6
const HESS = [:gn, :sr1, :bfgs, :hybrid_sr1, :hybrid_bfgs]

# Record the commit of the solver code used for the results
isempty(readchomp(`git -C $(@__DIR__) status --porcelain -- ../../src`)) ||
    error("Uncommitted changes in src/: commit them before running the experiments")
write(joinpath(RESULTS_DIR, "linear_constraints_commit.txt"),
      readchomp(`git -C $(@__DIR__) rev-parse HEAD`) * "\n")

for f in ["hs14", "hs22", "hs23", "hs32", "hs323", "hs42", "hs49", "hs50"]
    include(joinpath(@__DIR__, "..", "traulls_models", "$(f).jl"))
end
include(joinpath(@__DIR__, "models.jl"))

# Violation of the linear constraints lo ≤ A x ≤ hi at the point x
function linear_violation(name, x)
    A, lo, hi = LINCONS[name]
    Ax = A * x[1:size(A, 2)]
    return maximum(max.(lo .- Ax, Ax .- hi, 0.0))
end

out = joinpath(RESULTS_DIR, "linear_constraints.csv")
open(out, "w") do io
    println(io, "name,formulation,hessian,eps_active,status,objective,nouter_iter,ninner_iter,",
            "neval_residual,linear_violation,bound_violation,feasibility,n,nlincons,ncons")
end

for name in LIN_PROBLEMS, hess in HESS, (form, eps) in ((:E, 1e-3), (:E, 0.0), (:P, 1e-3))
    ctor = getfield(Main, Symbol(name, "_", form))
    model = ctor()
    res = traulls(ctor(); hessian_approx = hess, eps_active = eps,
                  max_iter = 500, max_inner_iter = 1000,
                  min_reltol_crit = OPT_CRIT, min_tol_feas = FEAS_CRIT)
    x = res.solution
    bound_violation = maximum(max.(model.xlow .- x, x .- model.xupp, 0.0))
    open(out, "a") do io
        @printf(io, "%s,%s,%s,%g,%s,%.17g,%d,%d,%d,%.3e,%.3e,%.3e,%d,%d,%d\n", name, form, hess,
                eps, res.status, res.objective / 2, res.counters.niter_outer,
                res.counters.niter_inner, res.counters.nres_eval, linear_violation(name, x),
                bound_violation, res.feasibility, model.n, model.nlincons, model.ncons)
    end
end
