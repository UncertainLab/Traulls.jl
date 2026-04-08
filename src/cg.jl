"""
    CG_status

Enum representing the termination status of the projected conjugate gradient method:

- `normal_exit`: The subproblem was solved successfully.
- `bound_hit`: The search direction hit a bound constraint.
- `negative_curvature`: Negative curvature was detected.
- `max_iter_reached`: The maximum number of iterations was reached.
"""
@enum CG_status normal_exit on_boundary negative_curvature max_iter_reached

""" 
    pcg!(b, H, w_l, w_u, fix_vars, κ_cg)

Approximately solves, w.r.t. `w` the subproblem:

`min 0.5 wᵀHw + wᵀb`

`s.t. Aw = 0`

`wᵢ = 0, i ∈ fix_vars`

`wₗ ≤ w ≤ wᵤ,`

using the projected conjugate gradient method.

Termination cases: 

- the norm of the preconditionned gradient has been reduced by a factor `κ_cg`

- direction of negative curvature is encountered (can happen when the Hessian is
 updated with SR1 formula)

- a conjugate direction goes beyond the feasible domain

- a maximum number of iterations  have been done (defined to be twice the number of free variables)

# Arguments

- `b`: Initial right handside vector

- `H`: Operator associated to the Hessian matrix

- `w_l`: Lower bounds for the variables

- `w_u`: Upper bounds for the variables

- `fix_vars`: Boolean vector indicating which variables are fixed
 
- `kappa_cg`: Tolerance parameter for convergence

- `atol`: Optionnal argument. Corresponds to absolute tolerance for negative
curvature detection (default: square root double relative precision)

# Returns

- `w`: The computed descent direction

- `status`: The termination status, encoded in the `CG_status` Enum (see [`CG_status`](@ref))
"""
function pcg!(
    b::AbstractVector{T},
    H::ALHessian,
    P::Projector{T},
    w::AbstractVector{T},
    w_l::AbstractVector{T},
    w_u::AbstractVector{T},
    r::AbstractVector{T},
    v::AbstractVector{T},
    p::AbstractVector{T},
    Hp::AbstractVector{T},
    kappa_cg::T;
    atol::T = sqrt(eps(T))) where T

    n = size(b,1)

    w .= 0.0

    r .= b
    mul!(v, P, r) # v ← Pr
    rtv = dot(r,v)
    p .= -v

    nrm_v = norm(v)
    tol_cg = nrm_v * min(kappa_cg, sqrt(nrm_v))
    tol_negcurve = atol

    iter = 1
    max_iter = 2*(nb_degrees_of_freedom(P))
    # approx_solved = abs(rtv) < tol_cg
    approx_solved = false
    neg_curvature = false
    outside_region = false

    while !approx_solved && !neg_curvature && !outside_region && iter <= max_iter

        # Form Hp and pᵀHp
        mul!(Hp,H,p)
        pHp = dot(p,Hp)

        if pHp <= tol_negcurve

            # Negative curvature 
            # Compute direction that stops at the feasible box and stop cg iterations
            neg_curvature = true

            if abs(pHp) > tol_negcurve # nonzero curvature
                gamma = factor_to_boundary(p,w,w_l,w_u,P)
                w .+= p .* gamma
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp
            gamma = factor_to_boundary(p, w, w_l, w_u, P)
            outside_region = alpha > gamma

            if outside_region
                # Next direction goes beyond feasible box
                # Compute direction that stops at the feasible box and stop cg
                # iterations
                w .+= p .* gamma
            else 
                # Update search and conjugate directions, evaluate convergence
                # criteria
                w .+= alpha .* p
                r .+= alpha .* Hp
                mul!(v,P,r) # v ← Pr
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-1, v, beta, p)         # p ← -v + βp
                rtv = rtv_next
                approx_solved = sqrt(rtv) < tol_cg  # ⟺ ||vₖ₊₁|| ≤ ε ||v₀||
                iter += 1
            end
        end
    end

    status = if approx_solved
        normal_exit
    elseif outside_region
        on_boundary
    elseif neg_curvature
        negative_curvature
    elseif iter > max_iter
        max_iter_reached
    end

    return status
end
