module Traulls

# Packages
using LinearAlgebra, Printf, Match, ForwardDiff, JuMP, HiGHS

import LinearAlgebra.mul!, LinearAlgebra.transpose

import Base.print, Base.println

# Abstract types

abstract type ALHessian{T} end

abstract type AbstractCnlsModel{T} end

abstract type Projector{T} end

# Include files

include("polyhedral_constraints.jl")
include("trust_region.jl")
include("cg.jl")
include("hessian.jl")
include("al_utils.jl")
include("workspace.jl")
include("cauchy.jl")
include("execution_metrics.jl")
include("model.jl")
include("print_info.jl")
include("solver.jl")

end 
