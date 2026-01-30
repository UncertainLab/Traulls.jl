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
    pcg(b, H, w_l, w_u, fix_vars, κ_cg)

Approximately solves, w.r.t. `w` the subproblem:

`min 0.5 wᵀHw + wᵀb`

`s.t. wᵢ = 0, i ∈ fix_vars`

` ` ` ` ` ` `wₗ ≤ w ≤ wᵤ,`

using the projected conjugate gradient method.

Termination cases: 

- the norm of the preconditionned gradient has been reduced by a factor `κ_cg`
- direction of negative curvature is encountered (can happen when the Hessian is updated with SR1 formula)
- a conjugate direction goes beyond the feasible domain
- a maximum number of iterations  have been done (defined to be twice the number of free variables)

# Arguments

- `b`: Initial right handside vector
- `H`: Operator associated to the Hessian matrix
- `w_l`: Lower bounds for the variables
- `w_u`: Upper bounds for the variables
- `fix_vars`: Boolean vector indicating which variables are fixed 
- `kappa_cg`: Tolerance parameter for convergence
- `atol`: Absolute tolerance for negative curvature detection (optional, default: square root double relative precision)

# Returns

- `w`: The computed descent direction
- `status`: The termination status, encoded in the `CG_status` Enum (see [`CG_status`](@ref))
"""
function pcg(
    b::Vector,
    H::ALHessian,
    w_l::Vector,
    w_u::Vector,
    fix_vars::BitVector,
    kappa_cg::Float64;
    atol::Float64 = sqrt(eps(Float64))) 

    
    n = size(b,1)

    # Buffers 
    w = zeros(n)
    r = zeros(n)
    v = zeros(n)
    
    # Form the preconditionner 
    # TODO: add preconditionning option
    free_vars = .!fix_vars
    P = Diagonal(free_vars) 

    r .= b
    v .= P*r
    rtv = dot(r,v)
    p = -v

    nrm_v = norm(v)
    tol_cg = nrm_v * min(kappa_cg, sqrt(nrm_v))
    tol_negcurve = atol

    iter = 1
    max_iter = 2*(n-count(fix_vars))
    # approx_solved = abs(rtv) < tol_cg
    approx_solved = false
    neg_curvature = false
    outside_region = false

    while !approx_solved && !neg_curvature && !outside_region && iter <= max_iter
        # println("[projected_cg] iter ", iter)
        Hp = H*p
        pHp = dot(p,Hp)

        if pHp <= tol_negcurve
            # Negative curvature 
            # Compute direction that stops at the feasible box and stop cg iterations
            neg_curvature = true
            if abs(pHp) > tol_negcurve # nonzero curvature
                gamma = factor_to_boundary(p,w,w_l,w_u,free_vars)
                w .+= p .* gamma
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp
            gamma = factor_to_boundary(p,w,w_l,w_u,free_vars)
            outside_region = alpha > gamma

            if outside_region
                # Next direction goes beyond feasible box
                # Compute direction that stops at the feasible box and stop cg iterations
                w .+= p .* gamma
            else 
                # Update search and conjugate directions, evaluate convergence criteria
                w .+= p .* alpha
                r .+= Hp .* alpha   
                v .= P*r
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-one(Float64), v, beta, p)         # p ← -v + βp
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

    return w, status
end

# Projected gradient method for the case with general linear equality constraints
#= Apply the projected conjugate gradient method to approximately solves the QP

`minₚ 0.5*pᵀHp + bᵀp`

`s.t. Ap = 0`

`pᵢ = 0, for i ∈ fixvars`

with early stopping if the iterations generate a direction of negative curvature
or that goes beyond implicit bounds on the free variables
=#                       
function pcg(
    b::Vector{T},
    H::ALHessian,
    P::SubspaceProjector{T},
    w_l::Vector{T},
    w_u::Vector{T},
    kappa_cg::T;
    atol::T = sqrt(eps(T))) where T
    
    n = size(b,1)

    # Buffers 
    w = zeros(n)
    r = zeros(n)
    v = zeros(n)
    
    r .= b
    mul!(v,P,r)
    rtv = dot(r,v)
    p = -v

    nrm_v = norm(v)
    tol_cg = nrm_v * min(kappa_cg, sqrt(nrm_v))
    tol_negcurve = atol

    iter = 1
    max_iter = 2*(n-count(fix_vars))
    # approx_solved = abs(rtv) < tol_cg
    approx_solved = false
    neg_curvature = false
    outside_region = false

    while !approx_solved && !neg_curvature && !outside_region && iter <= max_iter

        Hp = H*p
        pHp = dot(p,Hp)

        if pHp <= tol_negcurve
            # Negative curvature 
            # Compute direction that stops at the feasible box and stop cg iterations
            neg_curvature = true
            if abs(pHp) > tol_negcurve # nonzero curvature
                gamma = factor_to_boundary(p,w,w_l,w_u,free_vars)
                w .+= p .* gamma
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp
            gamma = factor_to_boundary(p,w,w_l,w_u,free_vars)
            outside_region = alpha > gamma

            if outside_region
                # Next direction goes beyond feasible box
                # Compute direction that stops at the feasible box and stop cg iterations
                w .+= p .* gamma
            else 
                # Update search and conjugate directions, evaluate convergence criteria
                w .+= p .* alpha
                r .+= Hp .* alpha   
                mul!(v,P,r)
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-one(Float64), v, beta, p)         # p ← -v + βp
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

    return w, status
end
