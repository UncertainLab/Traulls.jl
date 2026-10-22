# Performance profiles of the paper (files fig1 to fig14, which make up Figures 2, 3, 4 and 6),
# built from the result files with the code of profiles.ipynb and saved in the preprint
# folder. The file fig15 (Figure 5) is produced by residual_cost_analysis.jl.
#
# Usage, from the benchmark folder:
#     julia --project=. make_figures.jl

using Plots, BenchmarkProfiles, DataFrames, CSV

const RESULTS_DIR = joinpath(@__DIR__, "results")
const FIGURES_DIR = joinpath(@__DIR__, "..", "preprint", "figures")

savefigure(plt, name) = savefig(plt, joinpath(FIGURES_DIR, name * ".pdf"))

# Relative gap between the objective values reached by several solvers on an instance
relative_gap(obj_values) =
    (maximum(obj_values) - minimum(obj_values)) /
    (1 + max(abs(minimum(obj_values)), abs(maximum(obj_values))))

## Comparison of the Hessian approximations of TRAULLS (Figures 1 to 11)

traulls_variants = Dict{String, DataFrame}()
traulls_variants["Gauss-Newton"] = CSV.read(joinpath(RESULTS_DIR, "traulls_gn.csv"), DataFrame)
traulls_variants["BFGS"] = CSV.read(joinpath(RESULTS_DIR, "traulls_bfgs.csv"), DataFrame)
traulls_variants["SR1"] = CSV.read(joinpath(RESULTS_DIR, "traulls_sr1.csv"), DataFrame)
traulls_variants["Hybrid-BFGS"] = CSV.read(joinpath(RESULTS_DIR, "traulls_hybrid_bfgs.csv"), DataFrame)
traulls_variants["Hybrid-SR1"] = CSV.read(joinpath(RESULTS_DIR, "traulls_hybrid_sr1.csv"), DataFrame)

# Remove the instances on which the variants converge to different solutions
df_traulls = values(traulls_variants)
pb_names = first(df_traulls)[!, :name]
to_remove = [pb_names[i] for i in axes(pb_names, 1)
             if all(df -> df[i, :status] == "first_order_critical", df_traulls) &&
                relative_gap((df -> df[i, :objective]).(df_traulls)) > 1e-2]
println("Hessian approximations, instances removed: ", to_remove)

for v in values(traulls_variants)
    filter!(row -> !(row.name in to_remove), v)
end

function make_traulls_profile(metric, solvers, dict_lstyle; plot_title="")

    hess_variants = collect(keys(solvers))
    T = zeros(nrow(first(solvers).second), length(solvers))

    for (j, df) in enumerate(values(solvers))
        for (i, row) in enumerate(eachrow(df))
            T[i,j] = row[:status] == "first_order_critical" ? row[metric] : Inf
        end
    end

    performance_profile(PlotsBackend(), T, collect(keys(solvers)),
        palette=:tab10, linestyles = [dict_lstyle[s] for s in hess_variants],
        legend=:bottomright, xaxis = "τ (logscale)", yaxis = "ρ(τ)", title = plot_title)
end

lstyles = Dict("Gauss-Newton" => :solid,
    "BFGS" => :solid,
    "SR1" => :solid,
    "Hybrid-BFGS" => :dash,
    "Hybrid-SR1" => :dot)

for (metric, title, name) in [(:elapsed_time, "elapsed time", "fig1_traulls_variants_elapsed_time"),
                              (:ninner_iter, "inner iterations", "fig2_traulls_variants_ninner_iter"),
                              (:nouter_iter, "outer iterations", "fig3_traulls_variants_nouter_iter")]
    savefigure(make_traulls_profile(metric, traulls_variants, lstyles; plot_title = title), name)
end

function stratified_profile(metric, solvers, strat, dict_lstyle; plot_title="")

    hess_variants = collect(keys(solvers))
    T = zeros(length(strat), length(solvers))

    for (j, df) in enumerate(values(solvers))
        for (i, row) in enumerate(eachrow(filter(row -> row.name in strat, df)))
            T[i,j] = row[:status] == "first_order_critical" ? row[metric] : Inf
        end
    end

    performance_profile(PlotsBackend(), T, collect(keys(solvers)),
        palette=:tab10, linestyles = [dict_lstyle[s] for s in hess_variants],
        legend=:bottomright, xaxis = "τ (logscale)", yaxis = "ρ(τ)", title = plot_title,
        titlefontsize=8, guidefontsize=8)
end

# Strata on the squared norm of the residuals at the solution reached by Gauss-Newton; the
# objective recorded in the result files is 0.5 * ||r||^2
df_gn = traulls_variants["Gauss-Newton"]
strata = [("zero", filter(row -> 2*row.objective < 1e-8, df_gn)[!,:name], "||rₒₚₜ||² ≤ 10⁻⁸"),
          ("small", filter(row -> 1e-8 <= 2*row.objective < 1.0, df_gn)[!,:name], "10⁻⁸ ≤ ||rₒₚₜ||² < 1"),
          ("medium", filter(row -> 1.0 <= 2*row.objective < 1e2, df_gn)[!,:name], "1 ≤ ||rₒₚₜ||² < 100"),
          ("large", filter(row -> 1e2 <= 2*row.objective, df_gn)[!,:name], "||rₒₚₜ||² ≥ 100")]
println("Strata: ", [(s, length(names)) for (s, names, _) in strata])

variants_for_stratified = Dict(k => traulls_variants[k] for k in ["Gauss-Newton", "SR1", "Hybrid-SR1"])
lstyles = Dict("Gauss-Newton" => :solid, "SR1" => :dash, "Hybrid-SR1" => :dot)

for (first_fig, metric, title) in [(4, :elapsed_time, "elapsed time"),
                                   (8, :neval_residual, "residuals evaluations")]
    for (k, (stratum, names, range)) in enumerate(strata)
        plt = stratified_profile(metric, variants_for_stratified, names, lstyles;
            plot_title = title * " - $(length(names)) problems with " * range)
        savefigure(plt, "fig$(first_fig + k - 1)_$(stratum)_res_$(metric)")
    end
end

## Comparison of TRAULLS (Hybrid-SR1) with IPOPT (L-BFGS) and Percival (Figures 12 to 14)

df_traulls = CSV.read(joinpath(RESULTS_DIR, "traulls_hybrid_sr1.csv"), DataFrame)
df_ipopt_lbfgs = CSV.read(joinpath(RESULTS_DIR, "ipopt-lbfgs.csv"), DataFrame)
df_percival = CSV.read(joinpath(RESULTS_DIR, "percival.csv"), DataFrame)

function solvers_comparison_profile(metric, solvers, stats_traulls, stats_ipopt, stats_percival; plot_title="")
    T = zeros(nrow(stats_traulls), 3)

    for (i, row) in enumerate(eachrow(stats_traulls))
        T[i, 1] = row[:status] == "first_order_critical" ? row[metric] : Inf
    end

    for (i, row) in enumerate(eachrow(stats_ipopt))
        T[i, 2] = row[:status] == "first_order" ? row[metric] : Inf
    end

    for (i, row) in enumerate(eachrow(stats_percival))
        T[i, 3] = row[:status] == "first_order" ? row[metric] : Inf
    end

    performance_profile(PlotsBackend(), T, solvers,
        palette=:tab10, linestyles = [:dot, :solid, :dash],
        legend=:bottomright, xaxis = "τ (logscale)", yaxis = "ρ(τ)", title = plot_title)
end

# Remove the instances on which the three solvers converge to different solutions
to_remove = [df_traulls[i, :name] for i in 1:nrow(df_traulls)
             if df_traulls[i, :status] == "first_order_critical" &&
                df_ipopt_lbfgs[i, :status] == "first_order" && df_percival[i, :status] == "first_order" &&
                relative_gap((df -> df[i, :objective]).([df_traulls, df_ipopt_lbfgs, df_percival])) > 1e-2]
println("Solvers, instances removed: ", to_remove)

stats_traulls, stats_ipopt_lbfgs, stats_percival =
    (filter(row -> !(row.name in to_remove))).([df_traulls, df_ipopt_lbfgs, df_percival])

solvers = ["TRAULLS", "IPOPT-LBFGS", "Percival"]
for (metric, title, name) in [(:neval_grad, "gradient evaluations", "fig12_traulls_vs_others_neval_grad"),
                              (:elapsed_time, "elapsed time", "fig13_traulls_vs_others_elapsed_time"),
                              (:neval_residual, "residuals evaluations", "fig14_traulls_vs_others_neval_residual")]
    savefigure(solvers_comparison_profile(metric, solvers, stats_traulls, stats_ipopt_lbfgs,
                                          stats_percival; plot_title = title), name)
end
