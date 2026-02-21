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

# Constructor for Gauss-Newton approximation
# Takes jacobians and a penalty parameter as inputs
# Initializes the buffer vector to zero
#
#
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

    n = size(J,2)
    return GN(J,C,mu,zeros(n))
end

"""
    mul!(Hv, H::GN, v)

Overload the 3-argument `mul!` method to the type [`GN`](@ref) to compute
Hessian-vector without doing matrix-matrix multiplications.
"""

function mul!(Hv::Vector{T}, gn_op::GN{T}, v::Vector{T}) where T

    # Buffer vectors for intermediate computations
    temp = gn_op.temp

    # Reset result values to make sure it is zero
    Hv .= 0.0
    # JᵀJv term
    mul!(temp, gn_op.J, v) # form Jv
    mul!(Hv, gn.op.J', temp, 1, 1) # add JᵀJv to result Hv

    # μCᵀCv term
    mul!(temp, sr1_op.C, v) # form Cv
    mul!(Hv, sr1_op.C', temp, sr1_op.mu, 1) # add μCᵀCv to result Hv


    return
end

""" Base.:*(H::GN, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid
matrix-matrix multiplication
"""
function Base.:*(H::GN{T}, v::Vector{T}) where T
    Hv = Vector{T}(undef,size(v,1))
    mul!()
    return H.J' * (H.J*v) + H.C' * (H.mu*H.C*v)
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

# Constructor for `SR1` Hessian approximation.
# Takes jacobians and penalty parameters as inputs.
# Second order terms are initialized to `0`.
# Secant equations components and the buffer vector are
# set to `0`.

"""
    SR1(J,C,μ)

Constructor method for the [`GN`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the other
attributes to `0`.

** Arguments

* `J`: Jacobian matrix of the residuals

* `C`: Jacobian matrix of the nonlinear equality constraints

* `μ`: Penalty parameter
"""
function SR1(
    J::AbstractMatrix,
    C::AbstractMatrix,
    mu::Float64)

    n = size(J,2)

    return SR1(J,C,zeros(n,n),mu,zeros(n),zeros(n),zeros(n),zeros(n),zeros(n))
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

    # Buffer vectors for intermediate computations
    temp = sr1_op.temp

    # Reset result values to make sure it is zero
    Hv .= 0.0
    # JᵀJv term
    mul!(temp, sr1_op.J, v) # form Jv
    mul!(Hv, sr1.op.J', temp, 1, 1) # add JᵀJv to result Hv

    # μCᵀCv term
    mul!(temp, sr1_op.C, v) # form Cv
    mul!(Hv, sr1_op.C', temp, sr1_op.mu, 1) # add μCᵀCv to result Hv

    # Sv term
    mul!(Hv, sr1_op.S, v, 1, 1) # add Sv to result Hv

    return
end


"""
    update_sr1_second_order!(H::SR1)

Updates the second order terms of the Hessian approximation `H`.

Applies a structured SR1 update, with a safeguard check to prevent
the approximation to break down.
"""

function update_sr1_second_order!(sr1_op::SR1)

    # Tolerance for the skipping update safeguard
    eps_safeguard = sqrt(eps(T))

    # Vectors of the secant Equation Ss = y
    y = sr1_op.secant_rhs
    s = sr1_op.step

    # Buffer to store y - Ss
    ymSs = sr1_op.temp
    ymSs .= y
    mul!(ymSs, sr1_op.S, s, -1, 1) # Form ymSs = y - Ss

    # Apply update if denominator (y - Ss)ᵀs not too small
    denom = dot(s,ymSs)
    if abs(denom) > eps_safeguard * norm(s) * norm(ymSs)
        # Add (y - Ss)(y - Ss)ᵀ / (y - Ss)ᵀs to second order terms approximation
        mul!(sr1_op.S, ymSs, ymSs', 1/denom, 1)
    end

    return
end
