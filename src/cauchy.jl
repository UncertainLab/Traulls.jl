"""
    cauchy_step!(x,s,g,ℓ,u,sₗ,sᵤ,H,P,ℓ,d,Hd)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model
`q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the
projected gradient step `P(x-t*g) - x` with `P` denoting the projection over
`{s | Av = 0 and sₗ ≤ s ≤ sᵤ`, with `sₗ = max(-Δ, ℓ - x)`

This method finds the first local minimum of the quadratic model along the
projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

The associated Cauchy step is computed in place into vector `s`
Returns the `BitAbstractVector{T}` `fix_vars` that encodes the indices of active bounds
at the Cauchy point `x + s`.

Follows the procedure of algorithm 17.3.1 from Trust Regions Methods
(Conn, Gould and Toint, SIAM, 2000).

# Arguments

- `x`: current iterate
- `s`: buffer vector for the Cauchy step
- `g`: gradient of the augmented Lagrangian at current point
- `xlow`: lower bounds on the variables `x`
- `xupp`: upper bounds on the variables `x`
- `slow`: lower bounds on the step `s`
- `supp`: upper bounds on the step `s`
- `hess_op`: Operator for the Hessian approximation of type `ALHessian` at current point
- `proj_op`: [`SubspaceProjector`](@ref) operator onto tangent space of feasible directions
- `xlow`: lower bounds on the variables `x`
- `xupp`: upper bounds on the variables `x`
- `d`: buffer vector the the projected steepest direction
- `Hd`: Buffer vector to store Hessian-vector product

# On return

- `s` argument modified in place with components set to the Cauchy step
"""
function cauchy_step!(
    x::AbstractVector{T},
    s::AbstractVector{T},
    g::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T},
    slow::AbstractVector{T},
    supp::AbstractVector{T},
    hess_op::ALHessian{T},
    proj_op::Projector{T},
    d::AbstractVector{T},
    Hd::AbstractVector{T}) where T

    # Constants
    zeroT = T(0.0)
    eps_slope = 1e-10
    eps_curv = 1e-10

    # Initial projected steepest direction
    mul!(d, proj_op, -g) # d ← P[-g]

    # Fixed variables for the Cauchy step computation
    active, zero_dir = initial_fixed(x, d, xlow, xupp)
    fixed = vcat(active, zero_dir)

    # Update the projector operator and search direction
    !isempty(fixed) && set_active(proj_op, fixed)
    mul!(d, proj_op, -g)

    # Find first breakpoint
    s .= zeroT
    prev_tb = zeroT
    tb, idx = next_preakpoint(d, s, slow, supp, proj_op)


    # Form gᵀd and Hd
    gtd = dot(g,d)
    mul!(Hd, hess_op, d) # Hd ← H*d

    # Search for the first local minimum on the Cauchy path as long as bounds can become
    # active or breakpoints can be found

    found = false

    while !found && !saturated_subspace(proj_op) && !isempty(idx)

        # Compute slope and curvature
        phi_p = gtd + dot(s, Hd)
        phi_pp = dot(d, Hd)

        # Study the current interval [prev_tb, tb)
        delta_t = phi_pp > 0 ? -phi_p / phi_pp : zeroT
        l_interval = tb - prev_tb

        # Stop if slope positive or if slope is small and curvature is positive
        if phi_p > eps_slope || (abs(phi_p) <= eps_slope && phi_pp > eps_curv)
            found = true

        # Positive curvature and local minimum within current interval
        elseif phi_pp > 0 && delta_t < l_interval
            s .+= detla_t * d
            found = true

        # No local minimum in [prev_tb, tb)
        # Prepare for next interval
        else
            s .+= d .* l_interval # Accumulated step

            # Form next search direction
            set_active!(proj_op, idx)
            mul!(d, proj_op, -g)
            gdt = dot(g, d)
            mul!(Hd, hess_op, d)

            # Find next breakpoint
            prev_tb = tb
            gap, idx = next_breakpoint(d, s, slow, supp, proj_op)
            tb = prev_tb + gap
        end
    end

    # Remove zero directions from fixed variables
    set_free!(proj_op, zero_dir)

    return
end

# Finds the variables fixed during the Cauchy step computation
# It includes the variables lying at their upper or lower bound with direction moving out
# of the feasible region and variables whose associated component in the search direction
# is zero
# Uses square root of machine precision as relative tolerance to assert if bounds are active
# or if search direction is zero
#
# Returns the indices of the active variables and zero direction in separate integer arrays
function initial_fixed(
    x::AbstractVector{T},
    d::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T};
    epsrel::T = sqrt(eps(T))) where T

    active = []
    zero_dir = []

    # Components at bounds wiht direction moving out of the feasible region
    for i in axes(x,1)

        # Variable at lower bound with negative direction
        if x[i] <= xlow[i] + abs(xlow[i])*epsrel && d[i] < epsrel
            push!(active, i)

        # Variable at upper bound with positive direction
        elseif x[i] + abs(xupp[i])*epsrel >= xupp[i] && d[i] > -epsrel
            push!(active, i)

        # Variable between its bounds but with zero direction
        elseif abs(d[i]) <= epsrel
            push!(zero_dir, i)
        end
    end

    return active, zero_dir
end

# Finds the next breakpoint on the projected gradient path with a given set of variables
# already fixed
# Variables with breakpoint differing from a small quantity are associated to the same
# breakpoint.
#
# Returns the value of the gap between the previous and the next breakpoint

function next_breakpoint(
    d::AbstractVector{T},
    s::AbstractVector{T},
    slow::AbstractVector{T},
    supp::AbstractVector{T},
    proj_op::Projector{T};
    epsbp::T = 10*eps(T)) where T

    bp_value = Inf # current breakpoint value
    bp_idx = []    # indices of variables becoming active at breakpoint

    # TODO: filter the axes with free variables to get directly the iterator with the right
    # indices
    for i in axes(d,1)
        if !is_fixed(proj_op, i)
            bp_try = if d[i] < 0
                (slow[i] - s[i]) / d[i]
            elseif d[i] > 0
                (supp[i] - s[i]) / d[i]
            else
                Inf
            end

            also_bp = abs(bp_value - bp_try) < epsbp

            if also_bp
                push!(bp_idx, i)
            elseif !also_bp && bp_try < bp_value
                bp_value = bp_try
                bp_idx = [i]
            end
        end
    end

    return bp_value, bp_idx
end
