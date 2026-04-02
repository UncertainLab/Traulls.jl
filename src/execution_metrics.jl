# `@enum` type for the different termination status in Traulls
@enum CriticalityStatus first_order_critical feasible_non_critical infeasible_non_critical

# Structure whose fields store the number of function evaluations
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

# Structure storing the execution infos
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
