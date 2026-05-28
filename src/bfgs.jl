
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
# Second order terms are initialized with identity.
# The `small_res` parameter is set to `false`
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
    update_jacobians!(bfgs_op, J_new, C_new)

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
    if sty >= eps_skip * (1 + norm(s) * norm(y))

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
    update_jacobians!(hbfgs_op, J_new, C_new)

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
    if sty >= eps_skip * (1 + norm(s) * norm(y))

        # Rescale initial approximation at first iter
        # Assumes initial approximation is identity
        if first_iter
            scaling_factor = sty * (1 / dot(s, s))
            hbfgs_op.S .= T(0)
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


# Reset fields of the BFGS structure at the start of a new outer iteration
function reset_hessian!(
    H::BFGS{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    n = size(J0, 2)
    zero_T = zero(T)

    initial_second_order = Matrix{T}(I, n, n)

    update_jacobians!(H, J0, C0)
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

    update_jacobians!(H, J0, C0)
    H.mu = mu
    H.S .= initial_second_order
    H.step .= zero_T
    H.secant_rhs .= zero_T
    H.small_res = false
    return
end
