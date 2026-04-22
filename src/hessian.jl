"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation 
- `sr1`: second-order terms updates by a SR1 formula
"""

@enum HessianApprox gn sr1

"""
    GN <: ALHessian 

Mutable structure representing the Gauss-Newton approximation of the Augmented
Lagrangian Hessian

**Attributes**

* `J`: Jacobian of the residuals 

* `C`: Jacobian of the nonlinear constraints 

* `μ`: penalty parameter

* `temp`: buffer vector to avoid reallocations for intermediate
quantities involved when computing matrix-vector products

The resulting approximation is `H = JᵀJ + μCᵀC`.
"""
mutable struct GN{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    mu::T
    temp::AbstractVector{T}
end

"""
    GN(J,C,μ)

Constructor method for the [`GN`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the buffer
vector to zero.

* Arguments

- `J`: Jacobian matrix of the residuals

- `C`: Jacobian matrix of the nonlinear equality constraints

- `μ`: Penalty parameter
"""
function GN(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    m = size(J,1)
    p = size(C,1)
    return GN(copy(J),copy(C),mu,zeros(max(m,p)))
end

"""
    mul!(Hv, H::GN, v)

Overload the 3-argument `mul!` method to the type [`GN`](@ref) to compute
Hessian-vector without doing matrix-matrix multiplications.
"""

function mul!(Hv::Vector{T}, gn_op::GN{T}, v::Vector{T}) where T

    m = size(gn_op.J,1)
    p = size(gn_op.C,1)
    
    # Reset result values to make sure it is zero
    # Hv .= 0.0
    # JᵀJv term
    temp_Jv = view(gn_op.temp,1:m)
    mul!(temp_Jv, gn_op.J, v) # form Jv
    mul!(Hv, gn_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(gn_op.temp,1:p)
    mul!(temp_Cv, gn_op.C, v) # form Cv
    mul!(Hv, gn_op.C', temp_Cv, gn_op.mu, 1) # Hv ← Hv + μCᵀCv

    return
end

""" Base.:*(H::GN, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid
matrix-matrix multiplication
"""
function Base.:*(H::GN{T}, v::Vector{T}) where T
    Hv = Vector{T}(undef,size(v,1))
    mul!(Hv,H,v)

    return Hv
end

"""
    update_hessian!(H,J₊,C₊)

Updates the Gauss-Newton Hessian approximation `H` by modifiying the `J` and `C`
attributes with, respectively `J₊` and `C₊`.
"""
function update_hessian!(
    H::GN{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T}) where T
    
    H.J .= J_new
    H.C .= C_new
    return
end

"""
    reset_hessian!(H,J₀,C₀,μ₀)

Reset the Gauss-Newton approximation `H` by setting the `J`, `C` and `mu`
attributes to, respectively, `J₀`, `C₀` and μ₀.
"""
function reset_hessian!(
    H::GN{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T
    
    H.J .= J0
    H.C .= C0
    H.mu = mu0
    return
end

"""
    SR1 <: ALHessian

Mutable structure encoding the SR1 approximation of the augmented Lagrangian Hessian.

The approximation is of the form `H = JᵀJ + μCᵀC + S` where
`J` and `C` are available first order quantities, `μ` is the penalty parameter
 and `S` approximates the second order terms of the true Hessian.

Matrix `S` is updated iteratively by a SR1 formula derived from a structured
secant equation `Ss = y` where `s` is a step and right handside `y` is defined
by first order quantities.

** Attributes

* `J`: Jacobian of the residuals

* `C`: Jacobian of the nonlinear constraints

* `S`: approximation of the second order terms of the true Hessian

* `mu`: penalty parameter

* `step`: step of the current iteration

* `secant_rhs`: right handside of the structured secant equation

* `temp`: buffer vector to avoid reallocations for intermediate
quantities involved when computing matrix-vector products

"""
mutable struct SR1{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::Vector{T}
    secant_rhs::Vector{T}
    temp::Vector{T}
end

"""
    SR1(J,C,μ)

Constructor method for the [`GN`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the other
attributes to `0`.

**Arguments**

* `J`: Jacobian matrix of the residuals

* `C`: Jacobian matrix of the nonlinear equality constraints

* `μ`: Penalty parameter
"""
function SR1(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    (m,n) = size(J)
    p = size(C,1)

    return SR1(copy(J),
               copy(C),
               zeros(T,n,n),
               mu,
               zeros(T,n),
               zeros(T,n),
               zeros(T,max(n,m,p)))
end

""" Base.:*(H::SR1, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid
matrix-matrix multiplication
"""
function Base.:*(sr1_op::SR1{T}, v::Vector{T}) where T

    Hv = Vector{T}(undef,size(v,1))
    mul!(Hv, sr1_op, v)

    return Hv
end

"""
    mul!(Hv, H, v)

Overload the 3-argument `mul!` method to the type [`SR1`](@ref) to compute
Hessian-vector without doing matrix-matrix multiplications.
"""
function mul!(Hv::Vector{T}, sr1_op::SR1{T}, v::Vector{T}) where T

    m = size(sr1_op.J,1)
    p = size(sr1_op.C,1)

    # JᵀJv term
    temp_Jv = view(sr1_op.temp,1:m)
    mul!(temp_Jv, sr1_op.J, v) # form Jv
    mul!(Hv, sr1_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(sr1_op.temp,1:p)
    mul!(temp_Cv, sr1_op.C, v) # form Cv
    mul!(Hv, sr1_op.C', temp_Cv, sr1_op.mu, 1) # Hv ← Hv + μCᵀCv

    # Sv term
    mul!(Hv, sr1_op.S, v, 1, 1) # Hv ← Hv + Sv
    return
end

function update_hessian!(
    sr1_op::SR1{T}, 
    J_new::Matrix{T},
    C_new::Matrix{T},
    rx::Vector{T},
    cx::Vector{T},
    g::Vector{T},
    y::Vector{T},
    s::Vector{T}) where T 

    # Update components of the secant equation
    # Right handside ← (Jₖ₊₁ - Jₖ)ᵀrₖ₊₁ + (Cₖ₊₁ - Cₖ)ᵀ(yₖ + μₖcₖ₊₁)
    sr1_op.secant_rhs .= g - sr1_op.J' * rx .- sr1_op.C' * (y .+ sr1_op.mu.*cx)
    sr1_op.step .= s

    # Update Jacobians 
    sr1_op.J .= J_new
    sr1_op.C .= C_new

    # Compute Second order terms
    update_sr1_second_order!(sr1_op)

    return
end

"""
    update_sr1_second_order!(H::SR1)

Updates the second order terms of the Hessian approximation `H`.

Applies a structured SR1 update, with a safeguard check to prevent
the approximation to break down.
"""

function update_sr1_second_order!(sr1_op::SR1{T}) where T

    # Tolerance for the skipping update safeguard
    eps_safeguard = T(1e-8)

    # Vectors of the secant Equation Ss = y
    y = sr1_op.secant_rhs
    s = sr1_op.step

    # Buffer to store y - Ss
    ymSs = view(sr1_op.temp,1:size(s,1))
    ymSs .= y
    mul!(ymSs, sr1_op.S, s, -1, 1) # Form ymSs = y - Ss

    # Add (y - Ss)(y - Ss)ᵀ / (y - Ss)ᵀs to second order terms approximation
    # Update applied if denominator (y - Ss)ᵀs not too small
    denom = dot(s,ymSs)

    if abs(denom) > eps_safeguard * (1 + norm(s)*norm(ymSs))
        mul!(sr1_op.S, ymSs, ymSs', 1/denom, 1)
    end

    return
end

"""
    reset_hessian!(H,J₀,C₀,μ₀)

Reset the SR1 approximation `H` by setting the `J`, `C` and `mu`
attributes to, respectively, `J₀`, `C₀` and μ₀.

The second order terms in attribute `S` are set to `0`.
"""
function reset_hessian!(
    H::SR1{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    H.J .= J0
    H.C .= C0
    H.mu = mu0
    H.S .= 0

    return
end
