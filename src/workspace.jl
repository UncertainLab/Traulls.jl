# Workspace structure whose attributes are the buffers vectors involved into
# intermediate computations.
# Avoids doing unnecessary reallocations of memory throughout the execution.

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

# Reset the values of the field of `Workspace` to 0

function reset_workspace!(wrkspc::Workspace{T}) where T
    zero_T = T(0)

    wkrspc.x_prev .= zero_T
    wkrspc.rx_prev .= zero_T
    wkrspc.cx_prev .= zero_T
    wkrspc.proj_g .= zero_T
    wkrspc.step .= zero_T
    wkrspc.search_dir .= zero_T
    wkrspc.step_low .= zero_T
    wkrspc.step_upp .= zero_T
    wkrspc.hess_vec .= zero_T
    wkrspc.cg_rhs .= zero_T
    wkrspc.r .= zero_T
    wkrspc.v .= zero_T
    wkrspc.p .= zero_T

    return
end
