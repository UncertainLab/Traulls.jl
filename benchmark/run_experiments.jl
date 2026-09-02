# Load environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Include libraries
include("../src/Traulls.jl")
using .Traulls

using Plots, BenchmarkProfiles, DataFrames, CSV, Printf, ForwardDiff

using NLPModelsIpopt, Percival, SolverBenchmark, NLSProblems

