# `@enum` type for the different termination status in Traulls
@enum CriticalityStatus first_order_critical feasible_non_critical infeasible_non_critical

#
"""
    TraullsCounters

Structure whose fields store the number of function evaluations and iterations performed
during the execution of Traulls.

# Attributes

- `nres_eval`: number of evaluations of the residuals
- `ncons_eval`: number of evaluations of the nonlinear constraints
- `nobj_eval`: number of evaluations of the objective
- `njacres_eval`: number of evaluations of the residuals Jacobian
- `jaccons_eval`: number of evaluations of the nonlinear constraints Jacobian
- `nalobj_eval`: number of evaluations of the augmented Lagrangian objective
- `nalgrad_eval`: number of evaluations of the augmened Lagrangian gradient
- `niter_outer`: total number of outer iterations
- `niter_inner`: total number of inner iterations
"""
mutable struct TraullsCounters

    # Residuals and constraints evaluation
    nres_eval::Int
    ncons_eval::Int
    nobj_eval::Int
    njacres_eval::Int
    njaccons_eval::Int

    # AL related evaluations
    nalobj_eval::Int
    nalgrad_eval::Int

    # Number of iterations
    niter_outer::Int
    niter_inner::Int
end

# Constructor for TraullsCounters structure
function TraullsCounters()
    TraullsCounters(0, 0, 0, 0, 0, 0, 0, 0, 0)
end

# Resets the counters to 0
function reset!(counters::TraullsCounters)

    for f in fieldnames(TraullsCounters)
        setfield!(counters, f, 0)
    end

end

"""
    TraullsResults

Mutable structure gathering the informations about a solution computed by `Traulls` solver.

# Attributes

- `solution`: optimal solution of the optimization problem found by the solver
- `lagrange_mults`: vector of Lagrange multipliers associated to the equality constraints at
 the solution
- `objective`: value of the objective function, i.e. the squared sum of residuals at
solution
- `feasibility`: norm of the equality constraints at the solution
- `criticality`: value of the criticality measure, i.e. measure of optimality, at the
solution
- `counters`: structure of type `TraullsCounters` storing the number of evaluation functions
 and iterations performed
- `elapsed_time`: solving time measured during the execution in seconds
"""
mutable struct TraullsResults{T}

    solution::Vector{T}
    lagrange_mults::Vector{T}
    objective::T
    feasibility::T # norm nonlinear constraints
    criticality::T # norm projected lagrangian or reduced gradient
    status::CriticalityStatus

    # Metrics
    counters::TraullsCounters
    elapsed_time::T
end

import Base.print, Base.println

# Pretting printing for `TraullsResults`
function print(io::IO, results::TraullsResults)

    println(io, "\n")
    println(io, "Finished in $(results.counters.niter_outer) outer iterations
(with $(results.counters.niter_inner) total inner iterations)\n")
    println(io, "Squared sum of residuals............................: ",
            @sprintf("%.6e", results.objective))
    println(io, "Criticality measure.................................: ",
            @sprintf("%.6e", results.criticality))
    println(io, "Feasibility of equality constraints.................: ",
            @sprintf("%.6e", results.feasibility))
    println(io, "Termination status..................................: ", results.status)
    println(io, "\n")

    println(io, "Execution time......................................: ",
            @sprintf("%.3f seconds", results.elapsed_time))
    println(io, "Number of residuals evaluations.....................:",
            @sprintf(" %d", results.counters.nres_eval))
    println(io, "Number of residuals Jacobian evaluations............:",
            @sprintf(" %d", results.counters.njacres_eval))
    println(io, "Number of AL gradient evaluations...................:",
            @sprintf(" %d", results.counters.nalgrad_eval))
end

println(io::IO, results::TraullsResults) = print(io,results)
