#=
    sr1.jl

Structured SR1 and hybrid SR1 approximations of the augmented Lagrangian Hessian, where the
second order terms are updated with a SR1 formula.

Author(s): Pierre Borie
=#

"""
    SR1Formula <: SecantFormula

Marker type selecting the structured SR1 update of the second order terms of a
[`DenseSecantHessian`](@ref).
"""
struct SR1Formula <: SecantFormula end

"""
    SR1{T}

Alias of `DenseSecantHessian{T, SR1Formula}`, the SR1 approximation of the augmented
Lagrangian Hessian `H = JᵀJ + μCᵀC + S`.

Matrix `S` is initialized to `0` and updated iteratively by a SR1 formula derived from a 
NL2SOL style structured secant equation `Ss = y` where `s` is a step and right-hand side 
`y` is defined by first order quantities (see [`second_order_secant_update!`](@ref)).
"""
const SR1{T} = DenseSecantHessian{T, SR1Formula}

"""
    HybridSR1{T}

Alias of `Hybrid{T, SR1{T}}`, the hybrid switching between the Gauss-Newton and the
[`SR1`](@ref) approximations of the augmented Lagrangian Hessian.
"""
const HybridSR1{T} = Hybrid{T, SR1{T}}

"""
    SR1(J,C,μ)

Constructor method for the [`SR1`](@ref) approximation.

Takes jacobians and a penalty parameter as inputs and initializes the other
attributes to `0`.

# Arguments

- `J`: Jacobian matrix of the residuals
- `C`: Jacobian matrix of the nonlinear equality constraints
- `μ`: Penalty parameter
"""
SR1(J::AbstractMatrix{T}, C::AbstractMatrix{T}, mu::T) where T =
    DenseSecantHessian(SR1Formula, J, C, mu)

"""
    HybridSR1(J,C,μ)

Constructor method for the [`HybridSR1`](@ref) approximation.

Takes jacobians and a penalty parameter as inputs and initializes the other attributes to
`0`. The residuals are initially not considered small, i.e. `small_res` is set to `false`.
"""
HybridSR1(J::AbstractMatrix{T}, C::AbstractMatrix{T}, mu::T) where T =
    Hybrid(SR1(J, C, mu))

"""
    initialize_second_order!(H::SR1)

Set the second order terms `S` of the SR1 approximation `H` to `0`.
"""
function initialize_second_order!(H::SR1{T}) where T
    H.S .= zero(T)
    return
end

"""
    second_order_secant_update!(H::SR1)

Update the second order terms `S` of the SR1 approximation `H`.

Applies the structured SR1 update

`S₊ = S + (y - Ss)(y - Ss)ᵀ / (y - Ss)ᵀs,`

where `s` and `y` are the step and right-hand side of the secant equation stored in `H`.
As a safeguard to prevent the approximation from breaking down when the denominator `(y -
Ss)ᵀs` is too small, the update is skipped if

`(y - Ss)ᵀs ≤ ε * (1 + ||s|| * ||y - Ss||).`

# On return

Update `S₊` stored in `H`.
"""
function second_order_secant_update!(sr1_op::SR1{T}) where T

    # Tolerance for the skipping update safeguard
    eps_safeguard = T(1e-8)

    # Vectors of the secant Equation Ss = y
    y = sr1_op.secant_rhs
    s = sr1_op.step

    # Form y - Ss
    ymSs = view(sr1_op.temp, 1:size(s,1))
    ymSs .= y
    mul!(ymSs, sr1_op.S, s, -1, 1) # Form y - Ss
    denom = dot(s, ymSs)

    # Add (y - Ss)(y - Ss)ᵀ / (y - Ss)ᵀs to second order terms approximation
    # Update applied if denominator (y - Ss)ᵀs not too small

    if abs(denom) > eps_safeguard * (1 + norm(s) * norm(ymSs))
        mul!(sr1_op.S, ymSs, ymSs', 1/denom, 1)
        sr1_op.nupdates += 1
    end

    return
end
