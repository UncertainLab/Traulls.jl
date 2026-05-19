"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation 
- `sr1`: second-order terms updates by a SR1 formula
"""
@enum HessianApprox begin
    gn
    sr1
    bfgs
    hybrid_bfgs
    hybrid_sr1
end

const dict_hessians = Dict(:gn => gn,
                           :sr1 => sr1,
                           :bfgs => bfgs,
                           :hybrid_bfgs => hybrid_bfgs,
                           :hybrid_sr1 => hybrid_sr1)

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
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    temp::AbstractVector{T}
end

"""
    SR1(J,C,μ)

Constructor method for the [`SR1`](@ref) structure.

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

# Approximation of the AL Hessian with second order terms approximated with
# BFGS update formula
# Based on the NL2SOL-style secant equation
mutable struct BFGS{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    temp::Vector{T}
end

# Constructor for the BFGS structure
function BFGS(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    (m,n) = size(J)
    p = size(C, 1)

    initial_second_order = Matrix{T}(I, n, n)

    return BFGS(copy(J),
                copy(C),
                initial_second_order,
                mu,
                zeros(T,n),
                zeros(T,n),
                zeros(T, max(n,m,p)))
end



# Hybrid version of the structured BFGS update
mutable struct HybridBFGS{T<:Real} <:ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    small_res::Bool
    temp::AbstractVector{T}
end

# Constructor method for `HybridBFGS` struct
# Initializes the J,C attributes with jacobians evaluated at starting point
# Second order terms are initialized with identity scaled by the norm of the "augmented"
# residuals. The `small_res` parameter is set to `false`
# TODO: optimize the operations to make the computations less greedy and avoid redonduncy
function HybridBFGS(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    (m,n) = size(J)
    p = size(C, 1)

    initial_second_order = Matrix{T}(I, n, n)

    return HybridBFGS(copy(J),
                      copy(C),
                      initial_second_order,
                      mu,
                      zeros(T,n),
                      zeros(T,n),
                      false,
                      zeros(T, max(n,m,p)))
end

# Hybrid structured and scaled SR1 formula
# Uses the secant equation from Zhou and Chen and the same heuristic to detect small
# residuals problems and applies the SR1 update
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

# Constructor for the HybridSR1 struct
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

""" Base.:*(H::SR1, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid
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

# Overload the 3-argument `mul!` method to `BFGS` Hessian operator
function mul!(Hv::AbstractVector{T}, bfgs_op::BFGS{T}, v::AbstractVector{T}) where T

    m = size(bfgs_op.J, 1)
    p = size(bfgs_op.C, 1)

    # JᵀJv term
    temp_Jv = view(bfgs_op.temp, 1:m)
    mul!(temp_Jv, bfgs_op.J, v) # form Jv
    mul!(Hv, bfgs_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(bfgs_op.temp,1:p)
    mul!(temp_Cv, bfgs_op.C, v) # form Cv
    mul!(Hv, bfgs_op.C', temp_Cv, bfgs_op.mu, 1) # Hv ← Hv + μCᵀCv

    # Sv term
    mul!(Hv, bfgs_op.S, v, 1, 1) # Hv ← Hv + Sv
    return
end

# Overload the 3-argument `mul!` method to the `HybridBFGS` scheme
function mul!(Hv::AbstractVector{T}, hbfgs_op::HybridBFGS{T}, v::AbstractVector{T}) where T

    m = size(hbfgs_op.J, 1)
    p = size(hbfgs_op.C, 1)

    # JᵀJv term
    temp_Jv = view(hbfgs_op.temp,1:m)
    mul!(temp_Jv, hbfgs_op.J, v) # form Jv
    mul!(Hv, hbfgs_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(hbfgs_op.temp,1:p)
    mul!(temp_Cv, hbfgs_op.C, v) # form Cv
    mul!(Hv, hbfgs_op.C', temp_Cv, hbfgs_op.mu, 1) # Hv ← Hv + μCᵀCv

    # If non zero residuals, Hv ← Hv + Sv
    !hbfgs_op.small_res && mul!(Hv, hbfgs_op.S, v, 1, 1)
    return
end

# Overload the 3-argument `mul!` method to the `HybridSR1` scheme
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
    second_order_secant_update!(sr1_op)

    return
end

"""
    update_sr1_second_order!(H::SR1)

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
    mul!(ymSs, sr1_op.S, s, -1, 1) # Form y - τSs
    denom = dot(s, ymSs)

    # Add (y - τSs)(y - Ss)ᵀ / (y - τSs)ᵀs to second order terms approximation
    # Update applied if denominator (y - τSs)ᵀs not too small

    if abs(denom) > eps_safeguard * (1 + norm(s) * norm(ymSs))
        mul!(sr1_op.S, ymSs, ymSs', 1/denom, 1)
    end

    return
end

# Update the BFGS second order terms approximation
function update_hessian!(
    bfgs_op::BFGS{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T},
    first_iter::Bool) where T

    # Update components of the secant equation
    # Right handside ← (Jₖ₊₁ - Jₖ)ᵀrₖ₊₁ + (Cₖ₊₁ - Cₖ)ᵀ(yₖ + μₖcₖ₊₁)
    bfgs_op.secant_rhs .= g - bfgs_op.J' * rx .- bfgs_op.C' * (y .+ bfgs_op.mu.*cx)
    bfgs_op.step .= s

    # Update Jacobians
    bfgs_op.J .= J_new
    bfgs_op.C .= C_new

    # Compute Second order terms
    second_order_secant_update!(bfgs_op, first_iter)

    return

end

# Update the second_order terms of the Hessian approximation
function second_order_secant_update!(bfgs_op::BFGS{T}, first_iter::Bool) where T

    # Tolerance for skipping update
    eps_skip = T(1e-7)
    y = bfgs_op.secant_rhs
    s = bfgs_op.step

    sty = dot(s, y)

    # Update second order terms it sufficient curvature
    if sty >= eps_skip * norm(s) * norm(y)

        # Rescale initial approximation at first iter
        # Assumes initial approximation is identity
        if first_iter
            scaling_factor = sty * (1 / dot(s, s))
            for i in axes(bfgs_op.S, 1)
                bfgs_op.S[i, i] = scaling_factor
            end
        end

        # Buffer to store matrix-vector product
        Ss = view(bfgs_op.temp, 1:size(s, 1))
        mul!(Ss, bfgs_op.S, s)

        # Apply BFGS update
        mul!(bfgs_op.S, y, y', 1/sty, 1)           # S ← S + yyᵀ/sᵀy
        mul!(bfgs_op.S, Ss, Ss', -1/dot(s, Ss), 1) # S ← S - SssᵀS / sᵀSs

    end

    return
end



# Update the Hybrid-BFGS approximation
# Forms the secant equation right handside, new regularizatin factor and evaluate the small
# residuals heuristic
# Second order terms update is skipped if the residuals are considered zero
function update_hessian!(
    hbfgs_op::HybridBFGS{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T},
    cx_new::AbstractVector{T},
    fx_new::T,
    fx_prev::T,
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T},
    first_iter::Bool) where T


    # Constants
    eps_small_res = T(1//10) # Tjoa Biegler recommended value
    eps_red = 10 * eps(T)
    mu = hbfgs_op.mu

    # Evaluate small residuals heuristic
    delta_red = eps_red * max(1, abs(fx_prev))
    hbfgs_op.small_res = (fx_prev - fx_new + delta_red) > eps_small_res * fx_prev

    # Form secant equation components
    hbfgs_op.secant_rhs .= g .- hbfgs_op.J' * rx_new .- hbfgs_op.C' * (y .+ mu .* cx_new)
    hbfgs_op.step .= s

    # Update first order terms
    hbfgs_op.J .= J_new
    hbfgs_op.C .= C_new

    # Update second order terms
    second_order_secant_update!(hbfgs_op, first_iter)

    return
end


# Apply the `BFGS` secant update to the second order terms approximation
# TODO: try out the damped Powell update
function second_order_secant_update!(hbfgs_op::HybridBFGS{T}, first_iter::Bool) where T

    # Tolerance for skipping update
    eps_skip = T(1e-7)
    y = hbfgs_op.secant_rhs
    s = hbfgs_op.step

    sty = dot(s, y)

    # Update second order terms it sufficient curvature
    if sty >= eps_skip * norm(s) * norm(y)

        # Rescale initial approximation at first iter
        # Assumes initial approximation is identity
        if first_iter
            scaling_factor = sty * (1 / dot(s, s))
            for i in axes(hbfgs_op.S, 1)
                hbfgs_op.S[i, i] = scaling_factor
            end
        end

        # Buffer to store matrix-vector product
        Ss = view(hbfgs_op.temp, 1:size(s, 1))
        mul!(Ss, hbfgs_op.S, s)

        # Apply BFGS update
        mul!(hbfgs_op.S, y, y', 1/sty, 1)           # S ← S + yyᵀ/sᵀy
        mul!(hbfgs_op.S, Ss, Ss', -1/dot(s, Ss), 1) # S ← S - SssᵀS / sᵀSs

    end

    return
end


# Update the Hybrid SR1 Hessian approximation using the scaled secant approximation
# Small residuals are evaliuated using the curvature condition
# Second order terms are updated following the standard SR1 safeguard
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
    hsr1_op.J .= J_new
    hsr1_op.C .= C_new

    return
end

# TODO: merge the the update_sr1_second_order into one method
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

    n = size(J0, 2)
    zero_T = zero(T)

    H.J .= J0
    H.C .= C0
    H.mu = mu0
    H.S .= zero_T
    H.step .= zero_T
    H.secant_rhs .= zero_T

    return
end

# Reset fields of the BFGS structure at the start of a new outer iteration
function reset_hessian!(
    H::BFGS{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    n = size(J0, 2)
    zero_T = zero(T)

    initial_second_order = Matrix{T}(I, n, n)

    H.J .= J0
    H.C .= C0
    H.mu = mu0
    H.S .= initial_second_order
    H.step .= zero_T
    H.secant_rhs .= zero_T

    return
end

# Reset fields of the HybridBFGS structure at the start of a new outer iteration
function reset_hessian!(
    H::HybridBFGS{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu::T) where T

    n = size(J0, 2)
    zero_T = zero(T)
    initial_second_order = Matrix{T}(I, n, n)

    H.J .= J0
    H.C .= C0
    H.mu = mu
    H.S .= initial_second_order
    H.step .= zero_T
    H.secant_rhs .= zero_T
    H.small_res = false
    return
end

# Reset fields of the HybridSR1 structure at the start of a new outer iteration
# Current version: second order terms are are not reset
function reset_hessian!(
    H::HybridSR1{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu::T) where T

    n = size(J0, 2)
    zero_T = zero(T)

    H.J .= J0
    H.C .= C0
    H.S .= zero_T
    H.mu = mu
    H.step .= zero_T
    H.secant_rhs .= zero_T
    H.small_res = false
    return
end
