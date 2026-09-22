# Traulls execution, with JIT-free timings (see run_experiments_timed.jl)

# Parameters and tolerances
MAX_ITER = 500
MAX_INNER_ITER = 1000
OPT_CRIT = 1e-5
FEAS_CRIT = 1e-6


# Include problems definiton functions
files_prefix = vcat(fixed_dimensions_pb, variable_dimensions_pb)

for prefix in files_prefix
    include("traulls_models/$(prefix).jl")
end

# First execution to compile the solver core
traulls(hs65())

"""
    solve_timed(make_model, kwargs; label = "")

Solve the instance built by `make_model()` once as a warm-up, discard that
result, then solve it `N_REPEAT` more times and return the fastest result.

A fresh model is built for every call so that no state carries over from the
warm-up. Compilation inside a timed run, and any drift in the evaluation
counters across repeats, are recorded as diagnostics.
"""
function solve_timed(make_model, kwargs; label = "")

    # Warm-up: identical call, result discarded
    traulls(make_model(); kwargs...)

    best = nothing

    for k in 1:N_REPEAT
        GC.gc()
        timing = @timed traulls(make_model(); kwargs...)
        res = timing.value

        record_compile!(label, k, timing.compile_time)

        if best === nothing
            best = res
        else
            if res.counters.nres_eval != best.counters.nres_eval ||
               res.counters.niter_outer != best.counters.niter_outer
                push!(NONDET_WARNINGS, label)
            end
            res.elapsed_time < best.elapsed_time && (best = res)
        end
    end

    return best
end

# Dictionary to store the results obtained for each choice of Hessian approximation
traulls_stats = Dict{Symbol, Dict{String, Traulls.TraullsResults}}()
hessian_choices = [:gn, :sr1, :bfgs, :hybrid_sr1, :hybrid_bfgs]
for hessian in hessian_choices
    traulls_stats[hessian] = Dict{String, Traulls.TraullsResults}()
end

# Solve problems from Hock-Schittkowski
for id in fixed_dimensions_pb
    pb = Symbol(id)
    ctor = eval(pb)

    for hessian in hessian_choices
        kwargs = (hessian_approx = hessian,
                  max_iter = MAX_ITER, max_inner_iter = MAX_INNER_ITER,
                  min_reltol_crit = OPT_CRIT, min_tol_feas = FEAS_CRIT)

        traulls_stats[hessian][id] = solve_timed(() -> ctor(), kwargs;
                                                 label = "$(id) / $(hessian)")
    end

    @printf("\n===== %10s finished =====", id)
end

# Solve problems from Luksan-Vleck collection
for id in variable_dimensions_pb
    pb = Symbol(id)
    ctor = eval(pb)

    for n in lv_dim
        id_instance = id * @sprintf("_%d", n)

        for hessian in hessian_choices
            kwargs = (hessian_approx = hessian,
                      max_iter = MAX_ITER, max_inner_iter = MAX_INNER_ITER,
                      min_reltol_crit = OPT_CRIT, min_tol_feas = FEAS_CRIT)

            traulls_stats[hessian][id_instance] = solve_timed(() -> ctor(n), kwargs;
                                                              label = "$(id_instance) / $(hessian)")
        end

    @printf("\n===== %10s finished =====\n", id_instance)

    end
end

# Write results into CSV files

res_to_df(results) = DataFrame(name = name_instances,
    n = [size(results[pb].solution, 1) for pb in name_instances],
    elapsed_time = [results[pb].elapsed_time for pb in name_instances],
    objective = [results[pb].objective * (1/2) for pb in name_instances],
    neval_grad = [results[pb].counters.nalgrad_eval for pb in name_instances],
    neval_residual = [results[pb].counters.nres_eval for pb in name_instances],
    neval_jac_residual = [results[pb].counters.njacres_eval for pb in name_instances],
    nouter_iter = [results[pb].counters.niter_outer for pb in name_instances],
    ninner_iter = [results[pb].counters.niter_inner for pb in name_instances],
    status = [results[pb].status for pb in name_instances])

for hessian in hessian_choices
    CSV.write(joinpath(RESULTS_DIR, "traulls_" * String(hessian) * ".csv"),
              res_to_df(traulls_stats[hessian]))
end
