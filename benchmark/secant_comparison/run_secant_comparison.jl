# Structured secant equation of the paper against the standard secant equation, the
# Gauss-Newton term being subtracted (option `exact_secant`), on the 82 instances of the
# benchmark, for the dense secant approximations (remark of Section 2.4 of the paper).
# Iteration counts do not depend on timings, so that each instance is solved once.
#
# At every update of a secant approximation, the difference between the two right-hand
# sides, D_r + μD_c with D_r = Jᵀ(r₊ - r) - J₊ᵀJ₊s and D_c = Cᵀ(c₊ - c) - C₊ᵀC₊s, is
# recorded through a method of `update_hessian!` defined below for this purpose.
#
# Usage, from the benchmark folder (several hours, because of the instances on which the
# standard secant equation reaches the maximum number of outer iterations):
#     julia --project=. secant_comparison/run_secant_comparison.jl

include(joinpath(@__DIR__, "..", "..", "src", "Traulls.jl"))
using .Traulls
using LinearAlgebra, Printf, ForwardDiff, SparseArrays

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OPT_CRIT = 1e-5
const FEAS_CRIT = 1e-6
const HESS = [:sr1, :bfgs, :hybrid_sr1, :hybrid_bfgs]

# Record the commit of the solver code used for the results
isempty(readchomp(`git -C $(@__DIR__) status --porcelain -- ../../src`)) ||
    error("Uncommitted changes in src/: commit them before running the experiments")
write(joinpath(RESULTS_DIR, "secant_comparison_commit.txt"),
      readchomp(`git -C $(@__DIR__) rev-parse HEAD`) * "\n")

fixed_dimensions_pb = ["hs6", "hs13", "hs14", "hs16", "hs17", "hs18", "hs20",
"hs22", "hs23", "hs26", "hs27", "hs30", "hs31", "hs32", "hs42", "hs43", "hs46", "hs49",
"hs50", "hs57", "hs60", "hs61", "hs65", "hs70", "hs77", "hs79", "hs216", "hs227", "hs264",
"hs316", "hs323", "hs337", "hs344", "hs345", "hs354", "hs355", "hs372", "hs373", "hs394",
"hs395"]
lv_dim = [100, 500, 1000]
variable_dimensions_pb = ["BNST2", "BNST3", "lv501", "lv502", "lv503", "lv504", "lv511",
"lv512", "lv513", "lv514", "lv515", "lv516", "lv517", "lv518"]

for prefix in vcat(fixed_dimensions_pb, variable_dimensions_pb)
    include(joinpath(@__DIR__, "..", "traulls_models", "$(prefix).jl"))
end

instances = Tuple{String,Function}[]
for id in fixed_dimensions_pb
    ctor = getfield(Main, Symbol(id))
    push!(instances, (id, () -> ctor()))
end
for id in variable_dimensions_pb, n in lv_dim
    ctor = getfield(Main, Symbol(id))
    push!(instances, (id * @sprintf("_%d", n), () -> ctor(n)))
end

# Updates of the current run: (μ, ‖s‖, ‖D_r‖, ‖D_c‖, ‖D_r + μD_c‖)
const UPDATES = NTuple{5,Float64}[]

function Traulls.update_hessian!(
    H::Traulls.DenseSecantHessian{T},
    J_new::AbstractMatrix{T}, C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T}, cx_new::AbstractVector{T},
    fx_new::T, fx_prev::T,
    g::AbstractVector{T}, y::AbstractVector{T}, s::AbstractVector{T};
    kwargs...) where T

    rx_prev, cx_prev = kwargs[:rx_prev], kwargs[:cx_prev]
    D_r = H.J' * (rx_new - rx_prev) - J_new' * (J_new * s)
    D_c = H.C' * (cx_new - cx_prev) - C_new' * (C_new * s)
    push!(UPDATES, (H.mu, norm(s), norm(D_r), norm(D_c), norm(D_r + H.mu * D_c)))

    invoke(Traulls.update_hessian!,
           Tuple{Traulls.SecantHessian{T}, AbstractMatrix{T}, AbstractMatrix{T},
                 AbstractVector{T}, AbstractVector{T}, T, T,
                 AbstractVector{T}, AbstractVector{T}, AbstractVector{T}},
           H, J_new, C_new, rx_new, cx_new, fx_new, fx_prev, g, y, s; kwargs...)
end

# Updates of the structured runs, by decade of μ, for the steps with 1e-8 < ‖s‖ < 0.1
const BY_DECADE = Dict{Int, Vector{NTuple{4,Float64}}}()

runs = joinpath(RESULTS_DIR, "secant_comparison.csv")
open(runs, "w") do io
    println(io, "name,hessian,secant,status,objective,neval_residual,nouter_iter,ninner_iter,",
            "mu_max,nupdates")
end

for (name, ctor) in instances, hess in HESS, exact in (false, true)
    empty!(UPDATES)
    res = traulls(ctor(); hessian_approx = hess, exact_secant = exact,
                  max_iter = 500, max_inner_iter = 1000,
                  min_reltol_crit = OPT_CRIT, min_tol_feas = FEAS_CRIT)
    mu_max = isempty(UPDATES) ? NaN : maximum(first, UPDATES)
    open(runs, "a") do io
        @printf(io, "%s,%s,%s,%s,%.17g,%d,%d,%d,%.6g,%d\n", name, hess,
                exact ? "exact" : "structured", res.status, res.objective / 2,
                res.counters.nres_eval, res.counters.niter_outer, res.counters.niter_inner,
                mu_max, length(UPDATES))
    end
    if !exact
        for (mu, ns, dr, dc, diff) in UPDATES
            1e-8 < ns < 0.1 || continue
            push!(get!(BY_DECADE, round(Int, log10(mu)), NTuple{4,Float64}[]),
                  (dr / ns^2, dc / ns^2, diff / ns^2, mu * dc / (dr + mu * dc + eps())))
        end
    end
end

median_of(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]

open(joinpath(RESULTS_DIR, "secant_comparison_mu.csv"), "w") do io
    println(io, "log10_mu,nupdates,median_Dr_over_s2,median_Dc_over_s2,",
            "median_difference_over_s2,median_share_of_mu_Dc")
    for k in sort(collect(keys(BY_DECADE)))
        v = BY_DECADE[k]
        @printf(io, "%d,%d,%.4g,%.4g,%.4g,%.4f\n", k, length(v),
                (median_of([t[i] for t in v]) for i in 1:4)...)
    end
end
