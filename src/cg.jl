#=
    cg.jl

Projected conjugate gradient method used to refine the Cauchy step in the subspace of the
free variables.

Author(s): Pierre Borie
=#

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
    negative_dot
end

""" 
    pcg!(b,H,P,s,sₗ,sᵤ,Δ,r,v,p,Hp,κ_cg;ε_curv)

Refine the current step `s` with the projected conjugate gradient (CG) method.

# Problem setup

The quadratic model of the augmented Lagrangian at the current iterate is 

`q(s) = 0.5 sᵀHs + gᵀs`, 

Starting from the current step `s`, the CG iterations look for a descent direction `w` 
such that `s + w` further reduces the model, which amounts to approximately solve w.r.t. 
`w` the subproblem:

`min 0.5 wᵀHw + wᵀb`

`s.t. Aw = 0`

`wᵢ = 0, i ∈ fix_vars`

where:

- `b = Hs + g` is the gradient of the quadratic model at the current step `s`
- `A` is the matrix of the linear equality constraints (the constraint `Aw = 0` is
  absent when the problem only has bounds)
- `fix_vars` is the set of indices of the variables whose bounds are active at `x + s`.

The matrix `A` and the set `fix_vars` are stored in the projector `P`, so that `P` 
projects onto `{w | Aw = 0, wᵢ = 0 for i ∈ fix_vars}`.

To ensure the feasiblity of the total step `s + w`, the search direction is also subject 
to the implicit bounds

`sₗ - s ≤ w ≤ sᵤ - s,`

with `sₗ = max(-Δ, xₗ - x)` and `sᵤ = min(Δ, xᵤ - x)`. 

The successive CG updates are accumulated in place in `s`, which holds `s + w` on return.

# Termination cases

- the norm of the preconditionned gradient has been reduced by a factor `κ_cg`
- direction of negative curvature is encountered (can happen when the Hessian is
  updated with SR1 formula)
- a conjugate direction goes beyond the feasible domain (either a bound or the trust region)
- a maximum number of iterations have been done (defined to be twice the number of free variables)

# Arguments

- `b`: initial right-hand side vector `Hs + g`
- `H`: operator associated to the Hessian matrix
- `P`: projector operator onto `{w | Aw = 0, wᵢ = 0 for i ∈ fix_vars}`, used to compute the
  projected directions
- `s`: current step, starting point of the CG iterations
- `sₗ`: lower bounds for the step
- `sᵤ`: upper bounds for the step
- `Δ`: radius of the infinite norm trust region
- `κ_cg`: relative tolerance to assert convergence of the CG iterations
- `r`, `v`, `p`, `Hp`: Buffer vectors

# Keywords

- `ε_curv`: Absolute tolerance used to decide whether the Hessian curvature is negative. 
  Defaults to `1e-10`.

# On return

- argument `s` modified in place to store the total step `s + w`, obtained after the CG 
  iterations
- `status`: the termination status of the CG iterations, encoded as a [`CG_status`](@ref)
- `pred`: reduction of the model after taking the correction step `w`, i.e. `q(s + w) - 
  q(s)`
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
    mintol_cg::T;
    eps_curv::T = T(1e-10)) where T

    r .= b
    mul!(v, P, r) # v ← Pr
    rtv = dot(r, v)
    p .= -v
    pred = zero(T)

    if rtv < 0 return (negative_dot, pred) end

    # Set tolerance
    nrm_v = sqrt(rtv)
    eps_cg = mintol_cg * (1 + nrm_v)

    # Prepare for CG iterations
    iter = 1
    max_iter = 2*(nb_degrees_of_freedom(P))
    solved = nrm_v < eps_cg
    neg_curvature = false
    outside_boundary = false
    trust_region_hit = false


    while !solved && !neg_curvature && !outside_boundary && iter <= max_iter

        # Form Hp and pᵀHp
        mul!(Hp, H, p)
        pHp = dot(p,Hp)

        if pHp <= 0

            # Negative curvature 
            # Compute direction that stops at the feasible box and stop cg iterations
            neg_curvature = true

            if abs(pHp) > eps_curv
                # nonzero curvature to sill take a step
                gamma = factor_to_boundary(p, s, s_l, s_u, P)
                s .+= gamma .* p
                pred += -gamma * rtv + 0.5 * gamma^2 * pHp
            end
        else
            # Compute model minimizer over current direction
            rtv = dot(r, v)
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

                # Form next conjugate direction
                r .+= alpha .* Hp
                mul!(v, P, r)          # v ← Pr
                rtv_next = dot(r, v)
                beta = rtv_next / rtv
                axpby!(-1, v, beta, p) # p ← -v + βp

                # Evaluate termination criteria
                rtv = rtv_next

                if rtv < 0 return (negative_dot, pred) end

                optimal = sqrt(rtv) < eps_cg
                too_small = rtv + T(1) <= T(1)
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
