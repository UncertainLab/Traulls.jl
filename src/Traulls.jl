module Traulls

# Packages
using LinearAlgebra, JuMP, Ipopt, Printf, Match

import LinearAlgebra.mul!, LinearAlgebra.transpose

# Abstract types

abstract type TralcnllsData end

abstract type ALHessian end

abstract type AbstractCnlsModel end

abstract type PolyhedralConstraints end

abstract type Projector{T} end

# Utils
for f in ["misc", "polyhedral_constraints", "trust_region", "cg", "hessian", "al_utils"]
    include("./utils/$f.jl")
end

# Solvers
for f in ["boconls"]
    include("./solvers/$f.jl")
end


end
