# Break-even analysis on the cost of a residual evaluation based on the following model 
# of the total computation time
#
#     T(c) = elapsed_time + c * neval_residual
# 
# where `elasped_time` is the reported computation time and `c` is the additional cost of 
# a residual evaluation
#
# Usage, from the benchmark folder:
#     julia --project=. residual_cost_analysis.jl

using CSV, DataFrames, Printf, Plots

# Display the plot after runnign the file
const show_plot = false

const RESULTS_DIR = joinpath(@__DIR__, "results")
const FIGURE_PATH = joinpath(@__DIR__, "..", "preprint", "figures",
                             "fig15_breakeven_residual_cost.pdf")

# Same relative-gap heuristic as the one used to build the performance profiles
const GAP_TOL = 1e-2

# Strata on the squared norm of the residuals at the solution. The objective
# recorded in the CSV files is f = 0.5 * ||r||^2, hence the factor 2.
const STRATA = ["zero", "small", "medium", "large"]

function stratum(objective)
    r2 = 2 * objective
    r2 < 1e-8 && return "zero"
    r2 < 1.0 && return "small"
    r2 < 100.0 && return "medium"
    return "large"
end

same_solution(fa, fb) =
    (max(fa, fb) - min(fa, fb)) / (1 + max(abs(fa), abs(fb))) <= GAP_TOL

"""
    paired_results(reference, variant)

Join the two result files on the problem instance, keeping only the instances
on which both solvers agree on the solution in the sense of `same_solution`.
"""
function paired_results(reference::AbstractString, variant::AbstractString)
    ref = CSV.read(joinpath(RESULTS_DIR, reference), DataFrame)
    var = CSV.read(joinpath(RESULTS_DIR, variant), DataFrame)
    df = innerjoin(ref, var, on = [:name, :n], makeunique = true)
    df = df[same_solution.(df.objective, df.objective_1), :]
    df.stratum = stratum.(df.objective)
    return df
end

"""
    evaluation_ratios(df)

Geometric mean, per stratum, of the ratio of the evaluation counts of the
variant to those of the reference, together with the pairwise win, tie and
loss counts on the residual evaluations.
"""
function evaluation_ratios(df)
    rows = []
    for s in STRATA
        sub = df[df.stratum .== s, :]
        isempty(sub) && continue
        geomean(a, b) = exp(sum(log.(b ./ a)) / nrow(sub))
        push!(rows, (
            stratum = s,
            instances = nrow(sub),
            wins = count(sub.neval_residual_1 .< sub.neval_residual),
            ties = count(sub.neval_residual_1 .== sub.neval_residual),
            losses = count(sub.neval_residual_1 .> sub.neval_residual),
            residual = geomean(sub.neval_residual, sub.neval_residual_1),
            jacobian = geomean(sub.neval_jac_residual, sub.neval_jac_residual_1),
            gradient = geomean(sub.neval_grad, sub.neval_grad_1),
        ))
    end
    return DataFrame(rows)
end

"""
    breakeven_costs(df)

Break-even costs `c*`, in seconds, on the instances where a crossover exists.
"""
function breakeven_costs(df)
    dn = df.neval_residual .- df.neval_residual_1
    dt = df.elapsed_time_1 .- df.elapsed_time
    crossing = (dn .> 0) .& (dt .> 0)
    return sort((dt[crossing]) ./ (dn[crossing]))
end

"""
    win_fraction(df, c)

Fraction of instances on which the variant is faster than the reference once
every residual evaluation is charged an extra cost `c`, in seconds.
"""
function win_fraction(df, c)
    faster = (df.elapsed_time_1 .+ c .* df.neval_residual_1) .<
             (df.elapsed_time .+ c .* df.neval_residual)
    return count(faster) / nrow(df)
end

quantile_sorted(v, q) = isempty(v) ? NaN : v[max(1, ceil(Int, q * length(v)))]

function main()
    df = paired_results("traulls_gn.csv", "traulls_hybrid_sr1.csv")
    @printf("Instances retained: %d\n\n", nrow(df))

    println("Evaluation counts, Hybrid-SR1 relative to Gauss-Newton")
    println("(geometric mean of the ratio; a value below 1 favors Hybrid-SR1)\n")
    ratios = evaluation_ratios(df)
    @printf("%-8s %10s %6s %6s %8s %10s %10s %10s\n",
            "stratum", "instances", "wins", "ties", "losses",
            "residual", "jacobian", "gradient")
    for r in eachrow(ratios)
        @printf("%-8s %10d %6d %6d %8d %10.2f %10.2f %10.2f\n",
                r.stratum, r.instances, r.wins, r.ties, r.losses,
                r.residual, r.jacobian, r.gradient)
    end

    # Comparison on medium and large residuals instances
    nonzero = df[(df.stratum .== "medium") .| (df.stratum .== "large"), :]
    @printf("\nNonzero-residual instances (medium and large): %d\n", nrow(nonzero))

    costs = breakeven_costs(nonzero)
    @printf("\nBreak-even extra cost per residual evaluation (%d instances with a crossover)\n",
            length(costs))
    for (q, label) in [(0.25, "Q1"), (0.5, "median"), (0.75, "Q3")]
        @printf("  %-8s %8.4f ms\n", label, 1e3 * quantile_sorted(costs, q))
    end

    current = sort(nonzero.elapsed_time ./ nonzero.neval_residual)
    @printf("\nCurrent cost of one residual evaluation in the test set\n")
    @printf("  %-8s %8.4f ms\n", "median", 1e3 * quantile_sorted(current, 0.5))

    println("\nFraction of nonzero-residual instances on which Hybrid-SR1 is faster")
    grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for c in grid
        @printf("  c = %8.3f ms : %5.1f %%\n", 1e3 * c, 100 * win_fraction(nonzero, c))
    end

    # Figure showing win fraction as a function of the per-evaluation cost
    cgrid = vcat(0.0, exp10.(range(-6, 0, length = 200)))
    fractions = [100 * win_fraction(nonzero, c) for c in cgrid]
    median_cost = quantile_sorted(costs, 0.5)

    plt = plot(1e3 .* cgrid[2:end], fractions[2:end],
               xscale = :log10, linewidth = 2, legend = :bottomright,
               label = "Hybrid-SR1 faster",
               xlabel = "extra cost per residual evaluation (ms)",
               ylabel = "instances (%)", ylims = (0, 100))

    vline!(plt, [1e3 * median_cost], linestyle = :dash, linewidth = 2,
           label = "median break-even cost")
    hline!(plt, [100 * win_fraction(nonzero, 0.0)], linestyle = :dashdot,
           linewidth = 1, color = :gray, label = "measured times")

    show_fig && display(plt)
end

main()
