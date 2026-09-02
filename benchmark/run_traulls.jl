# Traulls execution

# Number of parameters for LV problems
lv_dim = [100, 500, 1000]

MAX_ITER = 500
MAX_INNER_ITER = 1000
OPT_CRIT = 1e-5
FEAS_CRIT = 1e-6


# Include problems definiton functions
# Hock Schittlowski collection
fixed_dimensions_pb = ["hs6", "hs13", "hs14", "hs16", "hs17", "hs18", "hs20", "hs22", 
"hs23", "hs26", "hs27", "hs30", "hs31", "hs32", "hs42", "hs43","hs57", "hs60", "hs61", 
"hs65", "hs70", "hs77", "hs79", "hs216", "hs227", "hs264", "hs316", "hs323", "hs337", "hs344", 
"hs345", "hs354", "hs355", "hs372", "hs373", "hs394", "hs395"]

# Luksan Vlcek
variable_dimensions_pb = ["BNST2", "BNST3", "lv501", "lv502", "lv503", "lv504", "lv511", 
"lv512", "lv513", "lv514", "lv515", "lv516", "lv517", "lv518"]

files_prefix = vcat(fixed_dimensions_pb, variable_dimensions_pb)

for prefix in files_prefix
    include("traulls_models/$(prefix).jl")
end



# First execution to compile solver
traulls(hs65())

# Dictionary to store the results obtained for each choice of Hessian approximation
traulls_stats = Dict{Symbol, Dict{String, Traulls.TraullsResults}}()
hessian_choices = [:gn, :sr1, :bfgs, :hybrid_sr1, :hybrid_bfgs]
for hessian in hessian_choices
    traulls_stats[hessian] = Dict{String, Traulls.TraullsResults}()
end

name_instances = Vector{String}([])
# Solve problems from Hock-Schittkowski
for id in fixed_dimensions_pb
    push!(name_instances, id)
    pb = Symbol(id)
    eval(pb)()
    
    for hessian in hessian_choices
        traulls_stats[hessian][id] = traulls(eval(pb)(); hessian_approx=hessian, 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)
    end

    @printf("\n===== %10s finished =====", id)
end

# Solve problems from Luksan-Vleck collection
for id in variable_dimensions_pb
    pb = Symbol(id)
    eval(pb)()
    
    for n in lv_dim
        id_instance = id * @sprintf("_%d", n)
        push!(name_instances, id_instance)
        
        for hessian in hessian_choices
            traulls_stats[hessian][id_instance] = traulls(eval(pb)(n); hessian_approx=hessian, 
            max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
            min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)
        end

    @printf("\n===== %10s finished =====\n", id_instance)

    end
end

# Write results into CSV files

res_to_df(results) = DataFrame(name = name_instances,
    n = [size(results[pb].solution, 1) for pb in name_instances],
    elapsed_time = [results[pb].elapsed_time for pb in name_instances],
    objective = [results[pb].objective for pb in name_instances],
    neval_grad = [results[pb].counters.nalgrad_eval for pb in name_instances],
    neval_residual = [results[pb].counters.nres_eval for pb in name_instances],
    neval_jac_residual = [results[pb].counters.njacres_eval for pb in name_instances],
    nouter_iter = [results[pb].counters.niter_outer for pb in name_instances],
    ninner_iter = [results[pb].counters.niter_inner for pb in name_instances],
    status = [results[pb].status for pb in name_instances])

for hessian in hessian_choices
    CSV.write("traulls_"*String(hessian)*".csv", res_to_df(traulls_stats[hessian]))
end