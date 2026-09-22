#=
    workspace.jl 

Structure and related methods for the memory allocation of the buffers vectors involved into intermediate computations.
    
Author(s): Pierre Borie 
=#

"""
    Workspace{T}

Structure whose attributes are buffer vectors allocated at the first iteration and then 
involved into intermediate computations.

# Fields

- `x_prev`: previous iterate 
- `rx_prev`: residuals evaluated at previous iterate
- `cx_prev`: nonlinear constraints evaluated at previous iterate
- `proj_g`: projection of the gradient onto the tangent space at current point
- `step`: step for the current iteration
- `search_dir`: search direction updated after conjugate gradient iterations
- `step_low`: lower bounds on the step during the current iteration 
- `step_upp`: upper bounds on the step during the current iteration
- `hess_vec`: Hessian-vector product
- `cg_rhs`: right hand-side of the quadratic program 
- `r`: residual vector (CG)
- `v`: projected residual (CG)
- `p`: conjugate direction (CG)
"""
mutable struct Workspace{T<:Real}

    # Current and previous point info
    x_prev::Vector{T}
    rx_prev::Vector{T}
    cx_prev::Vector{T}

    # Inner minimization related
    proj_g::Vector{T}
    step::Vector{T}
    search_dir::Vector{T}
    step_low::Vector{T}
    step_upp::Vector{T}
    hess_vec::Vector{T}
    cg_rhs::Vector{T}
    r::Vector{T}
    v::Vector{T}
    p::Vector{T}
end

# Constructor for `Workspace` structure
# n: numbers of variables
# m: number of residuals
# p: number of nonlinear constraints
function Workspace(T::DataType, n::Int, m::Int, p::Int)

    Workspace{T}(zeros(T,n),zeros(T,m),zeros(T,p),zeros(T,n),zeros(T,n),zeros(T,n),
              zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),
              zeros(T,n))
end

"""
    reset_workspace!(wrkspc)

Reset the values of the field of `wrkspc` to 0.
"""
function reset_workspace!(wrkspc::Workspace{T}) where T
    zero_T = T(0)

    wrkspc.x_prev .= zero_T
    wrkspc.rx_prev .= zero_T
    wrkspc.cx_prev .= zero_T
    wrkspc.proj_g .= zero_T
    wrkspc.step .= zero_T
    wrkspc.search_dir .= zero_T
    wrkspc.step_low .= zero_T
    wrkspc.step_upp .= zero_T
    wrkspc.hess_vec .= zero_T
    wrkspc.cg_rhs .= zero_T
    wrkspc.r .= zero_T
    wrkspc.v .= zero_T
    wrkspc.p .= zero_T

    return
end
