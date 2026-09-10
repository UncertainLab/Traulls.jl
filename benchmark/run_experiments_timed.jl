# Load environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Include libraries
include("../src/Traulls.jl")
using .Traulls

using DataFrames, CSV, Printf, ForwardDiff

using NLPModelsIpopt, Percival, SolverBenchmark, NLSProblems


const N_REPEAT = 3
const RESULTS_DIR = "results_timed"

mkpath(RESULTS_DIR)

# Diagnostics collected during the run, reported by `report_timing_diagnostics()`
const COMPILE_WARNINGS = Tuple{String,Int,Float64}[]
const NONDET_WARNINGS = String[]

"""
    record_compile!(label, repeat, compile_time)

Flag a timed run that still triggered compilation. On a correctly warmed-up
benchmark `compile_time` is exactly zero.
"""
function record_compile!(label, repeat, compile_time)
    compile_time > 0 && push!(COMPILE_WARNINGS, (label, repeat, compile_time))
    return nothing
end

function report_timing_diagnostics()
    println("\n", "="^75)
    if isempty(COMPILE_WARNINGS)
        println("Compilation check: OK - no timed run triggered compilation.")
    else
        @printf("Compilation check: %d timed run(s) still compiled:\n",
                length(COMPILE_WARNINGS))
        for (label, k, t) in COMPILE_WARNINGS
            @printf("    %-40s repeat %d : %.4f s\n", label, k, t)
        end
        println("  -> those timings are contaminated; the warm-up did not cover",
                " every code path.")
    end

    if isempty(NONDET_WARNINGS)
        println("Determinism check: OK - evaluation counters stable across repeats.")
    else
        @printf("Determinism check: counters varied across repeats on %d instance(s):\n",
                length(NONDET_WARNINGS))
        for label in unique(NONDET_WARNINGS)
            println("    ", label)
        end
    end
    println("="^75)
    return nothing
end

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
end

for id in variable_dimensions_pb
    for n in lv_dim
        id_instance = id * @sprintf("_%d", n)
        push!(name_instances, id_instance)
    end
end

# Run solvers
include("run_traulls_timed.jl")
include("run_percival_ipopt_timed.jl")

report_timing_diagnostics()
