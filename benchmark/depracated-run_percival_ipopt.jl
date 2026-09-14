# Running Percival.jl and Ipopt.jl
using NLPModelsIpopt, Percival, SolverBenchmark, NLSProblems

problems = [
    NLSProblems.hs06(), NLSProblems.hs13(), NLSProblems.hs14(), NLSProblems.hs16(), NLSProblems.hs17(),
    NLSProblems.hs18(), NLSProblems.hs20(), NLSProblems.hs22(), NLSProblems.hs23(), NLSProblems.hs26(),
    NLSProblems.hs27(), NLSProblems.hs30(), NLSProblems.hs31(), NLSProblems.hs32(), NLSProblems.hs42(), 
    NLSProblems.hs43(), NLSProblems.hs46(), NLSProblems.hs49(), NLSProblems.hs50(), NLSProblems.hs57(),
    NLSProblems.hs60(), NLSProblems.hs61(), NLSProblems.hs65(), NLSProblems.hs70(), NLSProblems.hs77(),
    NLSProblems.hs79(), tp216(), tp227(), tp264(), tp316(), tp323(), tp337(), tp344(), tp345(), tp354(),
    tp355(), tp372(), tp373(), tp394(), tp395(),
    NLSProblems.BNST2(100), NLSProblems.BNST2(500), NLSProblems.BNST2(1000),
    NLSProblems.BNST3(100), NLSProblems.BNST3(500), NLSProblems.BNST3(1000),
    LVcon501(100), LVcon501(500), LVcon501(1000),
    LVcon502(100), LVcon502(500), LVcon502(1000), 
    LVcon503(100), LVcon503(500), LVcon503(1000),
    LVcon504(100), LVcon504(500), LVcon504(1000),
    LVcon511(100), LVcon511(500), LVcon511(1000),
    LVcon512(100), LVcon512(500), LVcon512(1000),
    LVcon513(100), LVcon513(500), LVcon513(1000),
    LVcon514(100), LVcon514(500), LVcon514(1000),
    LVcon515(100), LVcon515(500), LVcon515(1000),
    LVcon516(100), LVcon516(500), LVcon516(1000),
    LVcon517(100), LVcon517(500), LVcon517(1000),
    LVcon518(100), LVcon518(500), LVcon518(1000)]


@assert length(problems) == length(name_instances) "problem mismatch"

# Ipopt keyword arguments
common = (tol=1e-5, max_iter = 1000, nlp_scaling_method="none",
        dual_inf_tol = Inf, constr_viol_tol = Inf,
	compl_inf_tol = Inf, acceptable_iter = 0, print_level=0)
	
solvers = Dict(:percival => model -> percival(model; inity = true,
                                              atol=1e-5,
					      rtol = 1e-5,
					      ctol=1e-6,
					      ω_min = 1e-5,
                                              max_time = 600.0,
					      subsolver_max_iter=1000),
               :ipopt => model -> ipopt(model; common...),
	       :ipopt_lbfgs => model -> ipopt(model; common..., hessian_approximation = "limited-memory")
	       )
stats = bmark_solvers(solvers, problems)

# Insert number of gradient evaluations for Ipopt
# Equals the number of jacobian evaluation minus 1
for ipopt_variant in [:ipopt, :ipopt_lbfgs]
    stats[ipopt_variant][!, :neval_grad] .= stats[ipopt_variant][!, :neval_jac_residual] .- 1
end

# Allign name instances
for solver in keys(solvers)
    stats[solver][!, :name] .= name_instances
end

# Write results into CSV files 

CSV.write("results/ipopt.csv", stats[:ipopt])
CSV.write("results/ipopt-lbfgs.csv", stats[:ipopt_lbfgs])
CSV.write("results/percival.csv", stats[:percival])