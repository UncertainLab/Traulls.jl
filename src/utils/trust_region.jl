
"""
    TrustRegion <: TralcnllsData

Mutable structure to represent a trust region constraint of the form `||s|| ≤ Δ` and its update parameters.

# Fields

- `radius`: radius of the trust region
- `increase_treshold`: threshold to detect very successful steps (scalar in `(0,1)`)
- `accept_treshold`: step acceptance threshold (scalar in `(0,1)`)
- `increase_factor`: factor to increase the radius (scalar greater than `1`)
- `decrease_factor`: factor to decrease the radius (scalar in `(0,1)`)
- `neg_ratio_factor`: factor to decrease the radius in case of negative ratio (scalar in `(0,1)`)
"""
mutable struct TrustRegion <: TralcnllsData
    radius::Float64
    accept_treshold::Float64 
    increase_treshold::Float64 
    decrease_factor::Float64 
    increase_factor::Float64 
    neg_ratio_factor::Float64
end

function TrustRegion(eta1::Float64, eta2::Float64, alpha1::Float64, alpha2::Float64, gamma2::Float64) 
    return TrustRegion(Inf, eta1, eta2, alpha1, alpha2, gamma2)
end

"""
    print_tr_header(tr;io)

Prints in a formated way the parameters of a trust region `tr` within input/output flow `io` (default is `stdout`).
"""
function print_tr_header(tr::TrustRegion;io::IO=stdout) 
    println(io,"\nTrust Region parameters")
    println(io, "Step acceptance treshold.............................: ", @sprintf("%5f", tr.accept_treshold))
    println(io, "Radius increase treshold.............................: ", @sprintf("%5f", tr.increase_treshold))
    println(io, "Radius increase factor...............................: ", @sprintf("%5f", tr.increase_factor))
    println(io, "Radius decrease factor...............................: ", @sprintf("%5f", tr.decrease_factor))
    println(io, "Negative ratio decrease factor.......................: ", @sprintf("%5f", tr.neg_ratio_factor))
    println(io,"\n")
end

""" 
    set_initial_radius!(tr,g;κ,p)

Set the field `radius` of the trust region `tr` to `κ*||g||ₚ`, where:

- `g` is the gradient of the objective function to minimize
- `κ` is a constant (default value to `0.1`)
- `||.||ₚ` denotes the  `p`-norm (default is the `∞`-norm)

This value correponds to the initial radius of an optimization process.
"""
function set_initial_radius!(
    tr::TrustRegion,
    g::Vector;
    kappa_radius::Float64=0.1,
    p::Float64=Inf) 

    tr.radius = kappa_radius * norm(g,p)
    return
end


"""
    accept_step(tr,ρ)

Asserts if the ratio `ρ` associated to the step computed in the current trust region `tr` is accepted or not.
"""
accept_step(tr::TrustRegion,rho::Float64)  = rho >= tr.accept_treshold 


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
    mx::Float64,
    mx_trial::Float64,
    pred::Float64)


    eps_ratio = 10*eps(Float64)
    delta_ratio = eps_ratio*max(1,abs(mx_trial))


    delta_ared = mx_trial - mx - delta_ratio
    delta_pred = pred - delta_ratio

    ratio = abs(delta_ared) < eps_ratio && abs(delta_pred) < eps_ratio ? 1.0 : delta_ared / delta_pred

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
    tr::TrustRegion,
    rho::Float64,
    norm_step::Float64)  

    tr.radius = if rho > tr.increase_treshold   # very successful step
        max(tr.increase_factor * norm_step, tr.radius) 
    
    elseif 0 < rho < tr.accept_treshold         # bad step
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
function factor_to_boundary(x::Vector, d::Vector, delta::Float64; atol::Float64=sqrt(eps(Float64))) 
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

"""
    factor_to_boundary(p,w,wₗ,wᵤ,free_vars;atol)

Computes the largest scalar `γ` such that `w + γp` stays in the box `[wₗ,wᵤ]`. The only components considered
are the one encoded in `free_vars`.
"""
function factor_to_boundary(
    p::Vector,
    w::Vector,
    w_l::Vector,
    w_u::Vector,
    free_vars::BitVector;
    atol::Float64 = sqrt(eps(Float64))) 

    gamma = Inf
    for i in axes(w,1)
        if free_vars[i]
            if p[i] < -atol 
                gamma = min(gamma, (w_l[i] - w[i]) / p[i])
            elseif p[i] > atol
                gamma = min(gamma, (w_u[i] - w[i]) / p[i])
            end
        end
    end
    return gamma
end

function check_stalling(
    s::Vector,
    x::Vector,
    delta::Float64;
    eps_rel = sqrt(eps(Float64)))
    
    # Check is the trial point x+s is undistinguishable (up to `eps_rel`) from current solution x
    
    small_step = true
    i = 1

    while small_step && i <= size(s,1)
        abs_xi = abs(x[i])
        eps_step = abs_xi > 0 ? eps_rel*abs_xi : eps_rel
        small_step = small_step && abs(s[i]) < eps_step
        i += 1
    end

    # Check if the trust region radius is too small 
    
    small_radius = delta <= eps_rel * norm(x,Inf)

    return small_step || small_radius
end
