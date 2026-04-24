
"""
    TrustRegion{T}

Mutable structure to represent a trust region constraint of the form `||s|| ≤ Δ` and its update parameters.

# Fields

- `radius`: radius of the trust region
- `increase_treshold`: threshold to detect very successful steps (scalar in `(0,1)`)
- `accept_treshold`: step acceptance threshold (scalar in `(0,1)`)
- `increase_factor`: factor to increase the radius (scalar greater than `1`)
- `decrease_factor`: factor to decrease the radius (scalar in `(0,1)`)
- `neg_ratio_factor`: factor to decrease the radius in case of negative ratio (scalar in `(0,1)`)
"""
mutable struct TrustRegion{T<:Real}
    radius::T
    accept_threshold::T
    increase_threshold::T
    decrease_factor::T
    increase_factor::T
    neg_ratio_factor::T
end

function TrustRegion(eta1::T, eta2::T, alpha1::T, alpha2::T, gamma2::T) where T

    # Check trust region parameters are valid
    !(0 < eta1 <= eta2 < 1 &&
    0 < alpha1 < 1 < alpha2) &&
    error("ArgumentError: trust regions parameters are not valid")

    TrustRegion(Inf, eta1, eta2, alpha1, alpha2, gamma2)
end

# Pretty printing of a trust region parameters
function print(io::IO, tr::TrustRegion)
    println(io, "Trust Region parameters")
    println(io, "Step acceptance threshold.............................: ", @sprintf("%5f", tr.accept_threshold))
    println(io, "Radius increase threshold.............................: ", @sprintf("%5f", tr.increase_threshold))
    println(io, "Radius increase factor...............................: ", @sprintf("%5f", tr.increase_factor))
    println(io, "Radius decrease factor...............................: ", @sprintf("%5f", tr.decrease_factor))
    println(io, "Negative ratio decrease factor.......................: ", @sprintf("%5f", tr.neg_ratio_factor))

end

println(io::IO, tr::TrustRegion) = print(io, "\n", tr)
"""
    set_initial_radius!(tr,g;κ,p)

Set the field `radius` of the trust region `tr` to `max(1, κ*||g||ₚ)`, where:

- `g` is the gradient of the objective function to minimize
- `κ` is a constant (default value to `0.1`)
- `||.||ₚ` denotes the  `p`-norm (default is the `∞`-norm)

This value correponds to the initial radius of an optimization process.
"""
function set_initial_radius!(
    tr::TrustRegion{T},
    g::AbstractVector{T};
    kappa_radius::T = T(1//10),
    p::T = T(Inf)) where T

    tr.radius = kappa_radius * norm(g,p)
    return
end


"""
    accept_step(tr,ρ)

Asserts if the ratio `ρ` associated to the step computed in the current trust region `tr` is accepted or not.
"""
accept_step(tr::TrustRegion{T},rho::T) where T = rho >= tr.accept_treshold


"""
    step_ratio(mx, mx_trial, pred)

Computes and returns the ratio of the actual reduction `mx_trial-mx` in the objective function over the reduction predicted by the model.

Note that for a quadratic model of the form `s ↦ 0.5*sᵀHs + gᵀs + mx`, the predicted reduction when taking step `s` is merely `0.5*sᵀHs + gᵀs`.

The value is computed to avoid roundoff errors when both reductions are very small, up to a tolerance slightly larger than double relative precision.

This method follows the procedure described in Trust Region Methods (Conn et. al, SIAM, 2000), section 17.4.2.

# Arguments

- `mx`: value of the objective function at current point
- `mx_trial`: value of the objective function at trial point (current point + step)
- `pred`: reduction predicted by the model

# On return

- `ratio`: Value of the ratio `(mx_trial-mx) / pred`
"""
function step_ratio(
    fx::T,
    fx_trial::T,
    pred::T) where T

    global debug
    global debug_io

    # Constants
    eps_ratio = 10 * eps(T)
    delta_ratio = eps_ratio * max(1, abs(fx))

    # Adjusted actual and predicted reductions to avoid roundoff errors
    delta_ared = fx_trial - fx - delta_ratio
    delta_pred = pred - delta_ratio

    debug && @printf(debug_io, "\n[step_ratio] ared = %.5e\n", fx_trial - fx)
    debug && @printf(debug_io, "\n[step_ratio] δ = %.5e ; δared = %.5e ; δpred = %.5e\n", delta_ratio, delta_ared, delta_pred)

    ratio = abs(delta_ared) < eps_ratio && abs(delta_pred) < eps_ratio ? 1.0 :
        delta_ared / delta_pred

    return ratio
end


"""
    update_radius!(tr,ρ,||s||)

Update the trust region radius according to the value of the radius `ρ` and using `||s||` the `∞`-norm of the step.

For clarity, we identify the fields of `tr` as 

- `η₁` for `tr.accept_treshold`
- `η₂` for `tr.increase_treshold`
- `α₁` for `tr.decrease_factor`
- `α₂` for `tr.increase_factor`
- `γᵦ` for `tr.neg_ratio_factor`
- `Δ` for the current trust region radius 

The `radius` field of `tr` is modified in the following way:

- if `ρ ≥ η₂` (very good step), set `max(α₂*||s||, Δ)`
- if `η₁ ≤ ρ < η₂` (good step), set `Δₖ`
- if `0 ≤ ρ < η₁` (bad step),  set `α₁*||s||`
- if `ρ < 0` (very bad step), set `min(α₁*||s||, γᵦ*Δ)`


"""
function update_radius!(
    tr::TrustRegion{T},
    rho::T,
    norm_step::T) where T

    tr.radius = if rho > tr.increase_threshold   # very successful step
        max(tr.increase_factor * norm_step, tr.radius)
    elseif 0 < rho < tr.accept_threshold         # bad step
        tr.decrease_factor * norm_step
    elseif rho < 0                              # Very bad step
        min(tr.decrease_factor * norm_step, tr.neg_ratio_factor * tr.radius)
    else                                        # successful step 
        tr.radius
    end

    return
end

"""
    factor_to_boundary(x,d,Δ)

Computes and returns the largest `α > 0` such that `||x + αd|| = Δ` where `||.||` denotes the euclidean norm.
"""
function factor_to_boundary(
    x::AbstractVector{T},
    d::AbstractVector{T},
    delta::T;
    atol::T=sqrt(eps(T))) where T

    xtd = dot(x,d)
    norm_d2 = dot(d,d)
    discr = 4*(xtd^2 - norm_d2* (dot(x,x) - delta^2))
    alpha = Inf

    if abs(discr) <= atol
        alpha = -xtd / (2*norm_d2)
    else
        root1 = (-xtd - sqrt(discr)) / (2*norm_d2)
        root2 = (-xtd + sqrt(discr)) / (2*norm_d2)
        alpha = max(root1,root2)
    end
    return alpha
end

# Returns `true` if the step `s` lies on the boundary of an infinite norm trust region,
# `false` if not

step_on_region(s::AbstractVector{T}, radius::T) where T =
    norm(s, Inf) + radius * sqrt(eps(T)) >= radius

# Returns true if the trust region radius is too small to make relevant progress
small_radius(x::AbstractVector{T}, radius::T) where T =
    radius <= 10 * eps(T) * (1 + norm(x, Inf))

# Returns `true` if two consecutive iterates have similar components and value of objective
# function, up to given relative tolerances
function check_stalling(
    s::Vector{T},
    x::Vector{T},
    fx::T,
    fx_next::T,
    accepted::Bool) where T

    global debug
    global debug_io

    eps_comp =1e-7
    eps_obj = 1e-10

    # Consecutive iterates are undistinguishable (up to `eps_comp`)
    small_step = all(abs.(s) .<= eps_comp .* (1 .+ abs.(x)))

    # Small variation of the objective after iteration
    small_obj_variation = abs(fx_next - fx) <= eps_obj * max(1, abs(fx))

    debug && println(debug_io, "[check_stalling] small_step : $(small_step) ; small_obj_variation : $(small_obj_variation)")

    return accepted && small_step && small_obj_variation
end
