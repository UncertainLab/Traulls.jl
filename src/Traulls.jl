module Traulls

# Packages
using LinearAlgebra, Printf, Match, ForwardDiff, JuMP, HiGHS

import LinearAlgebra.mul!, LinearAlgebra.transpose

import Base.print, Base.println

# Abstract types

abstract type ALHessian{T} end

abstract type AbstractCnlsModel{T} end

abstract type Projector{T} end

# Utils
for f in ["polyhedral_constraints", "trust_region", "cg", "hessian",
    "al_utils",  "workspace", "execution_metrics", "model", "print_info"]

    include("$f.jl")
end

# Solver files
for f in ["solver"]
    include("$f.jl")
end

end 
