"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation 
- `sr1`: second-order terms updates by a SR1 formula
"""
@enum HessianApprox begin
    gn
    sr1
    hybrid_bfgs
end

const dict_hessians = Dict(:gn => gn,
                           :sr1 => sr1,
                           :hybrid_bfgs => hybrid_bfgs)
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

# Hybrid structured BFGS formula
# Adapted from Zhou and Chen (2010) hybrid quasi-Newton method for nonlinear least-squares
# Based on the reformulation of the outer minimization problem  as a "primal-dual"
# least-squares objective
mutable struct HybridBFGS{T<:Real} <:ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    reg_factor::T
    small_res::Bool
end

# Constructor method for `HybridBFGS` struct
# Initializes the J,C attributes with jacobians evaluated at starting point
# Second order terms are initialized with identity scaled by the norm of the "augmented"
# residuals. The `small_res` parameter is set to `false`
# TODO: optimize the operations to make the computations less greedy and avoid redonduncy
function HybridBFGS(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T,
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T}) where T

    n = size(J, 2)
    norm_aug_res = norm(vcat(rx, sqrt(mu) * (cx + y * (1/mu))))
    initial_second_order = norm_aug_res .* Matrix{T}(I, n, n)

    return HybridBFGS(copy(J),
                      copy(C),
                      initial_second_order,
                      mu,
                      zeros(T,n),
                      zeros(T,n),
                      norm_aug_res,
                      false)
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

# Overload the 3-argument `mul!` method to the `HybridBFGS` scheme
# TODO: use more memory efficient computations
function mul!(Hv::AbstractVector{T}, hbfgs_op::HybridBFGS{T}, v::Vector{T}) where T
    # JᵀJv term
    Hv .= hbfgs_op.J' * (hbfgs_op.J * v)

    # μCᵀCv term
    Hv .+= hbfgs_op.mu * hbfgs_op.C' * (hbfgs_op.C * v)

    if !hbfgs_op.small_res
        Hv .+= hbfgs_op.S * v
    end

    return
end

"""
    update_hessian!(H, J₊, C₊)

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
    update_hessian!(H, J₊, C₊, r₊, c₊, g, y, s)

Updates the Hessian approxmation `H` by first modifying the `J` and `C` attributes with,
respectively `J₊` and `C₊`. Second order terms approximation is updated by a structured
formula based on the secant equation

`S₊s = [J₊ - J]ᵀr₊ + [C₊ - C]ᵀ[y + μc₊]`
"""
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

    # Form y - Ss
    ymSs = view(sr1_op.temp, 1:size(s,1))
    ymSs .= y
    mul!(ymSs, sr1_op.S, s, -1, 1) # Form y - τSs
    denom = dot(s, ymSs)

    # Add (y - τSs)(y - Ss)ᵀ / (y - τSs)ᵀs to second order terms approximation
    # Update applied if denominator (y - τSs)ᵀs not too small

    if abs(denom) > eps_safeguard * (1 + norm(s) * norm(ymSs))
        mul!(sr1_op.S, ymSs, ymSs', 1/denom, 1)
    end

    return
end

# Update the Hybrid-BFGS approximation
# Forms the secant equation right handside, new regularizatin factor and evaluate the small
# residuals heuristic
# Second order terms update is skipped if the residuals are considered zero
# TODO: Perform more memory efficient computations
function update_hessian!(
    hbfgs_op::HybridBFGS{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T},
    cx_new::AbstractVector{T},
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T}) where T


    eps_small_res = T(1e-6) # value used in Zhou, Chen paper
    mu = hbfgs_op.mu

    # Scaling factor : quotient between the norm of consecutive augmented residuals
    norm_new_aug_res = norm(vcat(rx_new, sqrt(mu) .* (cx_new + y .* (1/mu))))
    scaling_factor = norm_new_aug_res * (1 / hbfgs_op.reg_factor)

    # Secant equation right handside
    z = g - hbfgs_op.J' * rx_new .- hbfgs_op.C' * (y .+ hbfgs_op.mu.*cx_new)
    z *= scaling_factor

    # Evaluate small residuals heuristic
    zts = dot(z, s)
    small_res = zts < eps_small_res * (1 + dot(s, s))

    # Update second order terms if non zero residuals
    if !small_res
        Ss = hbfgs_op.S * s
        bfgs_update = z*z' .* (1 / zts) - Ss*Ss' .* (1 / dot(s, Ss))
        hbfgs_op.S .+= bfgs_update
    end

    # Update remaining structure fields
    hbfgs_op.J .= J_new
    hbfgs_op.C .= C_new
    hbfgs_op.step .= s
    hbfgs_op.secant_rhs .= z
    hbfgs_op.small_res = small_res
    hbfgs_op.reg_factor = norm_new_aug_res

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
    reset_hessian!(H,J₀,C₀,μ₀)

Reset the SR1 approximation `H` by setting the `J`, `C` and `mu`
attributes to, respectively, `J₀`, `C₀` and μ₀.

The second order terms in attribute `S` are set to `0`.

Test to see what happens when they are maintained
"""
function reset_hessian!(
    H::SR1{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    zero_T = zero(T)

    H.J .= J0
    H.C .= C0
    H.mu = mu0
    # H.S .= T(0)
    H.step .= zero_T
    H.secant_rhs .= zero_T

    return
end

# Reset fields of the HybridBFGS structure at the start of a new outer iteration
# Test to leave the second order terms unchanged but adapts the scaling factor
function reset_hessian!(
    H::HybridBFGS{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu::T,
    rx0::AbstractVector{T},
    cx0::AbstractVector{T},
    y::AbstractVector{T}) where T

    n = size(J0, 2)
    zero_T = zero(T)
    norm_aug_res = norm(vcat(rx0, sqrt(mu) * (cx0 + y * (1/mu))))
    initial_second_order = norm_aug_res .* Matrix{T}(I, n, n)

    H.J .= J0
    H.C .= C0
    H.mu = mu
    # H.S .= initial_second_order
    H.reg_factor = norm_aug_res
    H.step .= zero_T
    H.secant_rhs .= zero_T

    return
end
