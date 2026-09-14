# Load environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Include libraries
include("../src/Traulls.jl")
using .Traulls

using DataFrames, CSV, Printf, ForwardDiff

using NLPModelsIpopt, Percival, SolverBenchmark, NLSProblems

# Set up problem headers

# Hock Schittlowski collection
fixed_dimensions_pb = ["hs6", "hs13", "hs14", "hs16", "hs17", "hs18", "hs20", 
"hs22", "hs23", "hs26", "hs27", "hs30", "hs31", "hs32", "hs42", "hs43", "hs46", "hs49", 
"hs50", "hs57", "hs60", "hs61", "hs65", "hs70", "hs77", "hs79", "hs216", "hs227", "hs264", 
"hs316", "hs323", "hs337", "hs344", "hs345", "hs354", "hs355", "hs372", "hs373", "hs394", 
"hs395"]

# Luksan Vlcek
lv_dim = [100, 500, 1000]
variable_dimensions_pb = ["BNST2", "BNST3", "lv501", "lv502", "lv503", "lv504", "lv511", 
"lv512", "lv513", "lv514", "lv515", "lv516", "lv517", "lv518"]

name_instances = Vector{String}([])

for id in fixed_dimensions_pb
    push!(name_instances, id)
    pb = Symbol(id)
    
end

for id in variable_dimensions_pb
    for n in lv_dim
        id_instance = id * @sprintf("_%d", n)
        push!(name_instances, id_instance)
    end
end

# Run solvers 
include("run_traulls.jl")
include("run_percival_ipopt.jl")