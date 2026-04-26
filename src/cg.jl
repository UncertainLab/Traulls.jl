"""
    CG_status

Enum representing the termination status of the projected conjugate gradient method:

- `normal_exit`: The subproblem was solved successfully
- `on_boundary`: The search direction hits one of the bounds on the variables
- `on_trust_region`: The search direction stops at the trust region boundary
- `negative_curvature`: Negative curvature was detected
- `max_iter_reached`: The maximum number of iterations was reached
"""
@enum CG_status begin
    normal_exit
    on_boundary
    on_trust_region
    negative_curvature
    max_iter_reached
end

""" 
    pcg!(b, H, P, s,  s_l, s_u, radius, r, v, p, Hp, κ_cg; ε_curv)

Approximately solves, w.r.t. `w` the subproblem:

`min 0.5 wᵀHw + wᵀb`

`s.t. Aw = 0`

`wᵢ = 0, i ∈ fix_vars`

`wₗ ≤ w ≤ wᵤ,`

using the projected conjugate gradient method.

The search directions updates are accumulated in the current step `s`.

Termination cases: 

- the norm of the preconditionned gradient has been reduced by a factor `κ_cg`

- direction of negative curvature is encountered (can happen when the Hessian is
 updated with SR1 formula)

- a conjugate direction goes beyond the feasible domain (either a bound or the trust region)

- a maximum number of iterations have been done (defined to be twice the number of free variables)

# Arguments

- `b`: Initial right handside vector

- `H`: Operator associated to the Hessian matrix

- `P`: Projector operator to compute the projected directions

- `s_l`: Lower bounds for the step

- `s_u`: Upper bounds for the step

- `radius`: Radius of the infinite norm trust region
 
- `kappa_cg`: Tolerance parameter for convergence

- `eps_curv`: Optionnal tolerance to assert if the Hessian curvature is high enough
(default: 1e-10)

- `r`, `v`, `p`, `Hp`: Buffer vectors

# Returns

- step `s` modified in place
- `status`: The termination status, encoded in the `CG_status` Enum (see [`CG_status`](@ref))
- `pred`: value of the reduction of the model after taking step `s`
"""
function pcg!(
    b::AbstractVector{T},
    H::ALHessian,
    P::Projector{T},
    s::AbstractVector{T},
    s_l::AbstractVector{T},
    s_u::AbstractVector{T},
    radius::T,
    r::AbstractVector{T},
    v::AbstractVector{T},
    p::AbstractVector{T},
    Hp::AbstractVector{T},
    kappa_cg::T;
    eps_curv::T = T(1e-10)) where T

    r .= b
    mul!(v, P, r) # v ← Pr
    rtv = dot(r,v)
    p .= -v


    eps_cg = kappa_cg * sqrt(rtv) # ϵ ← κ * ||v₀||

    # Prepare for CG iterations
    iter = 1
    max_iter = 2*(nb_degrees_of_freedom(P))
    # approx_solved = abs(rtv) < tol_cg
    solved = false
    neg_curvature = false
    outside_boundary = false
    trust_region_hit = false
    pred = zero(T)

    while !solved && !neg_curvature && !outside_boundary && iter <= max_iter

        # Form Hp and pᵀHp
        mul!(Hp, H, p)
        pHp = dot(p,Hp)

        if pHp <= 0

            # Negative curvature 
            # Compute direction that stops at the feasible box and stop cg iterations
            neg_curvature = true

            if abs(pHp) > eps_curv # nonzero curvature to sill take a step
                # Compute direction that stops at the feasible box and stop cg
                # iterations
                gamma = factor_to_boundary(p, s, s_l, s_u, P)
                s .+= gamma .* p
                pred += -gamma * rtv + 0.5 * gamma^2 * pHp
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp
            gamma = factor_to_boundary(p, s, s_l, s_u, P)
            outside_boundary = alpha > gamma

            if outside_boundary
                # Next direction goes beyond feasible box
                # Compute direction that stops at the feasible box and stop cg
                # iterations
                s .+= gamma .* p
                pred += -gamma * rtv + 0.5 * gamma^2 * pHp
                # Check if the step lies at the trust region boundary
                trust_region_hit = step_on_region(s, radius)
            else 
                # Increment step and predicted reduction
                s .+= alpha .* p
                pred -= 0.5 * alpha * rtv

                # Form next conjugate directions
                r .+= alpha .* Hp
                mul!(v, P, r) # v ← Pr
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-1, v, beta, p) # p ← -v + βp
                rtv = rtv_next

                # Evaluate termination criteria
                nrm_v = sqrt(rtv)
                optimal = nrm_v < eps_cg
                too_small = nrm_v + T(1) <= T(1)
                solved = optimal || too_small

                iter += 1
            end
        end
    end

    status = if solved
        normal_exit
    elseif trust_region_hit
        on_trust_region
    elseif outside_boundary
        on_boundary
    elseif neg_curvature
        negative_curvature
    elseif iter > max_iter
        max_iter_reached
    end

    return status, pred
end
