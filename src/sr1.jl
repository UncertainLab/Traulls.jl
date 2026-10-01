#=
    sr1.jl

Structured SR1 and hybrid SR1 approximations of the augmented Lagrangian Hessian.

Author(s): Pierre Borie
=#

"""
    SR1 <: ALHessian

Mutable structure encoding the SR1 approximation of the augmented Lagrangian Hessian.

The approximation is of the form `H = JᵀJ + μCᵀC + S` where
`J` and `C` are available first order quantities, `μ` is the penalty parameter
 and `S` approximates the second order terms of the true Hessian.

Matrix `S` is updated iteratively by a SR1 formula derived from a structured
secant equation `Ss = y` where `s` is a step and right-hand side `y` is defined
by first order quantities.

# Attributes

- `J`: Jacobian of the residuals
- `C`: Jacobian of the nonlinear constraints
- `S`: approximation of the second order terms of the true Hessian
- `mu`: penalty parameter
- `step`: step of the current iteration
- `secant_rhs`: right handside of the structured secant equation
- `temp`: buffer vector to avoid reallocations for intermediate
  quantities involved when computing matrix-vector products

"""
mutable struct SR1{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    temp::AbstractVector{T} # buffer vector for intermediate computations
end

"""
    SR1(J,C,μ)

Constructor method for the [`SR1`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the other
attributes to `0`.

# Arguments

- `J`: Jacobian matrix of the residuals
- `C`: Jacobian matrix of the nonlinear equality constraints
- `μ`: Penalty parameter
"""
function SR1(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    (m, n) = size(J)
    p = size(C,1)

    return SR1(copy(J),
               copy(C),
               zeros(T, n, n),
               mu,
               zeros(T,n),
               zeros(T,n),
               zeros(T,max(n,m,p)))
end

"""
    HybridSR1 <: ALHessian

Mutable structure encoding the hybrid SR1 approximation of the augmented Lagrangian
Hessian.

As for [`SR1`](@ref), the approximation is of the form `H = JᵀJ + μCᵀC + S`, where `S`
approximates the second order terms of the true Hessian and is updated by an SR1 formula
derived from a structured secant equation.

The second order terms are only used when the problem is detected to have non-small
residuals. This detection relies on a heuristic based on the relative reduction of the
augmented Lagrangian objective over the last step. When the residuals are deemed small,
the approximation reduces to the Gauss-Newton one `H = JᵀJ + μCᵀC`, but `S` keeps being
updated.

# Attributes

- `J`: Jacobian of the residuals
- `C`: Jacobian of the nonlinear constraints
- `S`: approximation of the second order terms of the true Hessian
- `mu`: penalty parameter
- `step`: step of the current iteration
- `secant_rhs`: right-hand side of the structured secant equation
- `small_res`: `true` if the residuals are deemed small, in which case `S` is not used in
  the Hessian-vector products
- `temp`: buffer vector to avoid reallocations for intermediate quantities involved when
  computing matrix-vector products
"""
mutable struct HybridSR1{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    small_res::Bool
    temp::AbstractVector{T}
end

"""
    HybridSR1(J,C,μ)

Constructor method for the [`HybridSR1`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the other attributes to
`0`. The residuals are initially not considered small, i.e. `small_res` is set to `false`.

# Arguments

- `J`: Jacobian matrix of the residuals
- `C`: Jacobian matrix of the nonlinear constraints
- `μ`: Penalty parameter
"""
function HybridSR1(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    (m,n) = size(J)
    p = size(C, 1)

    HybridSR1(copy(J),
              copy(C),
              zeros(T, n, n),
              mu,
              zeros(T,n),
              zeros(T,n),
              false,
              zeros(T, max(n, m, p)))
end

""" 
    Base.:*(H::SR1, v)

Overload the `*` operator to the type [`SR1`](@ref) in order to avoid
matrix-matrix multiplication
"""
function Base.:*(sr1_op::SR1{T}, v::AbstractVector{T}) where T

    Hv = Vector{T}(undef,size(v,1))
    mul!(Hv, sr1_op, v)

    return Hv
end

"""
    mul!(Hv, H, v)

Overload the 3-argument `mul!` method to the type [`SR1`](@ref) to compute
Hessian-vector without doing matrix-matrix multiplications.
"""
function mul!(Hv::Vector{T}, sr1_op::SR1{T}, v::AbstractVector{T}) where T

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


"""
    mul!(Hv, H::HybridSR1, v)

Overload the 3-argument `mul!` method to the type [`HybridSR1`](@ref) to compute the
Hessian-vector product `Hv` without doing matrix-matrix multiplications.

The product is `(JᵀJ + μCᵀC)v`, to which `Sv` is added only if the residuals are not
deemed small, i.e. if attribute `small_res` is `false`. The result is stored in `Hv`.
"""
function mul!(Hv::AbstractVector{T}, hsr1_op::HybridSR1{T}, v::AbstractVector{T}) where T
    m = size(hsr1_op.J, 1)
    p = size(hsr1_op.C, 1)

    # JᵀJv term
    temp_Jv = view(hsr1_op.temp,1:m)
    mul!(temp_Jv, hsr1_op.J, v) # form Jv
    mul!(Hv, hsr1_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(hsr1_op.temp,1:p)
    mul!(temp_Cv, hsr1_op.C, v) # form Cv
    mul!(Hv, hsr1_op.C', temp_Cv, hsr1_op.mu, 1) # Hv ← Hv + μCᵀCv

    # If non zero residuals, Hv ← Hv + Sv
    !hsr1_op.small_res && mul!(Hv, hsr1_op.S, v, 1, 1)

    return
end


"""
    update_hessian!(H, J₊, C₊, r₊, c₊, g, y, s)

Updates the Hessian approxmation `H` by first modifying the `J` and `C` attributes with,
respectively `J₊` and `C₊`. Second order terms approximation is updated by a structured
formula based on the secant equation

`S₊s = [J₊ - J]ᵀr₊ + [C₊ - C]ᵀ[y + μc₊]`
"""
function update_hessian!(
    sr1_op::SR1{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
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
    update_jacobians!(sr1_op, J_new, C_new)

    # Compute Second order terms
    second_order_secant_update(sr1_op)
    return
end

"""
    second_order_secant_update!(H::SR1)

Updates the second order terms of the Hessian approximation `H`.

Applies a structured SR1 update, with a safeguard check to prevent
the approximation to break down.
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
    end

    return
end


"""
    update_hessian!(H::HybridSR1, J₊, C₊, r₊, c₊, f₊, f, g, y, s)

Update the hybrid SR1 Hessian approximation `H` after an accepted step `s`.

First, the small residuals heuristic is evaluated: the residuals are deemed small if the
relative reduction of the augmented Lagrangian objective satisfies

`f - f₊ + δ > ε f,`

where `ε = 0.1` (value recommended by Tjoa and Biegler) and `δ` is a small
safeguard proportional to the machine precision. The result is stored in attribute
`small_res`.

Then, the second order terms `S` are updated by the SR1 formula derived from the
structured secant equation

`S₊s = [J₊ - J]ᵀr₊ + [C₊ - C]ᵀ[y + μc₊],`

with the standard SR1 safeguard that skips the update when its denominator is too small
(see [`second_order_secant_update!`](@ref)). Finally, the `J` and `C` attributes are
replaced by, respectively, `J₊` and `C₊`.

# Arguments

- `H`: Hessian approximation to update
- `J₊`: Jacobian of the residuals at the new point
- `C₊`: Jacobian of the nonlinear constraints at the new point
- `r₊`: residuals at the new point
- `c₊`: nonlinear constraints at the new point
- `f₊`: augmented Lagrangian objective value at the new point
- `f`: augmented Lagrangian objective value at the previous point
- `g`: gradient of the augmented Lagrangian at the new point
- `y`: Lagrange multipliers
- `s`: accepted step
"""
function update_hessian!(
    hsr1_op::HybridSR1{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T},
    cx_new::AbstractVector{T},
    fx_new::T,
    fx_prev::T,
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T}) where T


    # Constants
    eps_small_res = T(1//10) # Tjoa Biegler recommended value
    eps_red = 10 * eps(T)
    mu = hsr1_op.mu

    # Evaluate small residuals heuristic
    delta_red = eps_red * max(1, abs(fx_prev))
    hsr1_op.small_res = (fx_prev - fx_new + delta_red) > eps_small_res * fx_prev

    # Secant equation right handside
    hsr1_op.secant_rhs .= g .- hsr1_op.J' * rx_new .- hsr1_op.C' * (y .+ hsr1_op.mu.*cx_new)
    hsr1_op.step .= s

    # Update second order terms
    second_order_secant_update!(hsr1_op)

    # Update first order terms
    update_jacobians!(hsr1_op, J_new, C_new)

    return
end

# TODO: merge the the update_sr1_second_order into one method
"""
    second_order_secant_update!(H::HybridSR1)

Update the second order terms `S` of the hybrid SR1 approximation `H`.

Applies the structured SR1 update

`S₊ = S + (y - Ss)(y - Ss)ᵀ / (y - Ss)ᵀs,`

where `s` and `y` are the step and right-hand side of the secant equation stored in `H`.
As a safeguard to prevent the approximation from breaking down, the update is skipped if
the denominator `(y - Ss)ᵀs` is too small.
"""
function second_order_secant_update!(hsr1_op::HybridSR1{T}) where T

    # Tolerance for the skipping update safeguard
    eps_safeguard = T(1e-8)

    # Vectors of the secant Equation Ss = y
    y = hsr1_op.secant_rhs
    s = hsr1_op.step

    # Form y - Ss
    ymSs = view(hsr1_op.temp, 1:size(s,1))
    ymSs .= y
    mul!(ymSs, hsr1_op.S, s, -1, 1) # Form y - Ss
    denom = dot(s, ymSs)

    # Add (y - Ss)(y - σSs)ᵀ / (y - Ss)ᵀs to second order terms approximation
    # Update applied if denominator (y - Ss)ᵀs not too small

    if abs(denom) > eps_safeguard * (1 + norm(s) * norm(ymSs))
        mul!(hsr1_op.S, ymSs, ymSs', 1/denom, 1)
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

    n = size(J0, 2)
    zero_T = zero(T)

    update_jacobians!(H, J0, C0)
    H.mu = mu0
    H.S .= zero_T
    H.step .= zero_T
    H.secant_rhs .= zero_T

    return
end

"""
    reset_hessian!(H::HybridSR1, J₀, C₀, μ₀)

Reset the hybrid SR1 approximation `H` at the start of a new outer iteration, by setting
the `J`, `C` and `mu` attributes to, respectively, `J₀`, `C₀` and `μ₀`.

The second order terms in attribute `S`, the step and the secant right-hand side are set
to `0`, and the residuals are no longer considered small (`small_res` is set to `false`).
"""
function reset_hessian!(
    H::HybridSR1{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu::T) where T

    n = size(J0, 2)
    zero_T = zero(T)

    update_jacobians!(H, J0, C0)
    H.S .= zero_T
    H.mu = mu
    H.step .= zero_T
    H.secant_rhs .= zero_T
    H.small_res = false
    return
end
