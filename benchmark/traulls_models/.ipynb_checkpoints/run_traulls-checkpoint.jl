using Pkg

Pkg.activate(".")

using ForwardDiff, BenchmarkTools, BenchmarkPlots, StatsPlots, Printf, Dates

traulls_path = homedir()*"/.julia/dev/Traulls/src/"
include(traulls_path*"Traulls.jl")
using .Traulls

# TO MODIFY BEFORE EXECUTING
PUT_RESULTS_IN_FILE = true
lv_dim = [100, 500, 1000]

MAX_ITER = 500
MAX_INNER_ITER = 1000
OPT_CRIT = 1e-5
FEAS_CRIT = 1e-6


# Include problems definiton functions
# Hock Schittlowski collection
fixed_dimensions_pb = ["hs57", "hs65", "hs355", "hs354", "hs355", "hs394", "hs395"]

# Luksan Vlcek
variable_dimensions_pb = ["lv501", "lv502", "lv503", "lv504", "lv511", "lv512", "lv513", "lv514", "lv515",
             "lv516", "lv517", "lv518"]

files_prefix = vcat(fixed_dimensions_pb, variable_dimensions_pb)

for prefix in files_prefix
    include("$(prefix).jl")
end

# First execution to compile everything
traulls(hs65())
traulls(hs65(); hessian_approx=:sr1)
traulls(hs65(); hessian_approx=:hybrid_sr1)
traulls(hs65(); hessian_approx=:hybrid_bfgs)
traulls(hs65(); hessian_approx=:bfgs)

# Dictionaries to store the results
dict_gn = Dict{String, Traulls.TraullsResults}()
dict_sr1_hybrid = Dict{String, Traulls.TraullsResults}()
dict_sr1 = Dict{String, Traulls.TraullsResults}()
dict_bfgs_hybrid = Dict{String, Traulls.TraullsResults}()
dict_bfgs = Dict{String, Traulls.TraullsResults}()


# Solve problems from Hock-Schittkowski
for id in fixed_dimensions_pb
    pb = Symbol(id)

    dict_gn[id] = traulls(eval(pb)(); 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

    dict_sr1[id] = traulls(eval(pb)(); hessian_approx=:sr1, 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

    dict_bfgs[id] = traulls(eval(pb)(); hessian_approx = :bfgs,
    max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
    min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

    dict_sr1_hybrid[id] = traulls(eval(pb)(); hessian_approx=:hybrid_sr1, 
    max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
    min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

    dict_bfgs_hybrid[id] = traulls(eval(pb)(); hessian_approx=:hybrid_bfgs, 
    max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
    min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

    println("=====" * id * " finished =====")

end

# Solve problems from Luksan-Vleck collection
for id in variable_dimensions_pb
    pb = Symbol(id)

    for n in lv_dim
        id_instance = id * @sprintf("_%d", n)

        dict_gn[id_instance] = traulls(eval(pb)(n); 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

        dict_sr1[id_instance] = traulls(eval(pb)(n); hessian_approx=:sr1, 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

        dict_bfgs[id_instance] = traulls(eval(pb)(n); hessian_approx = :bfgs,
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

        dict_sr1_hybrid[id_instance] = traulls(eval(pb)(n); hessian_approx=:hybrid_sr1, 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

        dict_bfgs_hybrid[id_instance] = traulls(eval(pb)(n); hessian_approx=:hybrid_bfgs, 
        max_iter = MAX_ITER, max_inner_iter=MAX_INNER_ITER,
        min_reltol_crit=OPT_CRIT, min_tol_feas=FEAS_CRIT)

        println("=====" * id_instance * " finished =====")
    

    end
end

if PUT_RESULTS_IN_FILE
    results_io = open("../results/results_traulls_jopt.jl", "w")
    println(results_io, "gn_results = ", dict_gn)
    println(results_io, "sr1_results= ", dict_sr1)
    println(results_io, "hybrid_sr1_results = ", dict_sr1_hybrid)
    println(results_io, "bfgs_results = ", dict_bfgs)
    println(results_io, "hybrid_bfgs_results = ", dict_bfgs_hybrid)
    close(results_io)
end