module Traulls

# Packages
using LinearAlgebra, JuMP, Ipopt, Printf, Match

import LinearAlgebra.mul!, LinearAlgebra.transpose

# Abstract types

abstract type TralcnllsData end

abstract type ALHessian{T} end

abstract type AbstractCnlsModel end

abstract type PolyhedralConstraints end

abstract type Projector{T} end

# Utils
for f in ["misc", "polyhedral_constraints", "trust_region", "cg", "hessian", 
    "al_utils"]

    include("$f.jl")
end

# Solvers
for f in ["boconls_solver", "traulls_solver"]
    include("$f.jl")
end


end 
