#=
    cauchy.jl

Computation of the Cauchy step along the projected gradient path, which provides the
starting point for the approximate solution of the trust region subproblem.

Author(s): Pierre Borie
=#

"""
    cauchy_step!(x,s,g,ℓ,u,sₗ,sᵤ,hess_op,proj_op,d,Hd)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model
`q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the
projected gradient step `P(x-t*g) - x` with `P` denoting the projection over
`{s | Av = 0 and sₗ ≤ s ≤ sᵤ`, with `sₗ = max(-Δ, ℓ - x)` and 
`sᵤ = min(Δ, u - x)`

This method finds the first local minimum of the quadratic model along the
projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

The associated Cauchy step is computed in place into vector `s`
Returns the `BitAbstractVector{T}` `fix_vars` that encodes the indices of active bounds
at the Cauchy point `x + s`.

Follows the procedure of algorithm 17.3.1 from Trust Regions Methods
(Conn, Gould and Toint, SIAM, 2000).

# Arguments

- `x`: current iterate
- `s`: buffer vector to store the Cauchy step
- `g`: gradient of the augmented Lagrangian at current point
- `ℓ`: lower bounds on the variables `x`
- `u`: upper bounds on the variables `x`
- `sₗ`: lower bounds on the step `s`
- `sᵤ`: upper bounds on the step `s`
- `hess_op`: Operator for the Hessian approximation of type `ALHessian` at current point
- `proj_op`: [`SubspaceProjector`](@ref) operator onto tangent space of feasible directions
- `d`: buffer vector to store the projected steepest directions
- `Hd`: Buffer vector to store Hessian-vector products

# On return

- `pred`: reduction of the quadratic model obtained after taking the Cauchy step
- argument `s` stores the resulting Cauchy step
- argument `proj_op` corresponds to the projector operator onto the null space of the
  active constraints (linear equalities + active bounds)
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
    mul!(d, proj_op, g, -one(T), zero(T)) # d ← P[-g]

    # Fixed variables for the Cauchy step computation
    active, zero_dir = initial_fixed(x, d, xlow, xupp)
    fixed = vcat(active, zero_dir)

    # Update the projector operator and search direction
    !isempty(fixed) && set_active!(proj_op, fixed)
    mul!(d, proj_op, g, -one(T), zero(T)) # d ← P[-g]

    # Find first breakpoint
    s .= zeroT
    prev_tb = zeroT
    tb, idx = next_breakpoint(d, s, slow, supp, proj_op)


    # Form gᵀd and Hd
    gd = dot(g,d)
    mul!(Hd, hess_op, d) # Hd ← H*d

    # Search for the first local minimum on the Cauchy path as long as bounds can become
    # active or breakpoints can be found

    found = false
    pred = zeroT
    tc = zeroT
    i = 1
    while !found && !saturated_subspace(proj_op) && !isempty(idx)

        # Compute slope and curvature
        phi_p = gd + dot(s, Hd)
        phi_pp = dot(d, Hd)

        # Study the current interval [prev_tb, tb)
        delta_t = phi_pp > 0 ? -phi_p / phi_pp : zeroT
        l_interval = tb - prev_tb

        # Stop if slope positive or if slope is small and curvature is positive
        if phi_p > eps_slope || (abs(phi_p) <= eps_slope && phi_pp > eps_curv)
            found = true

        # Positive curvature and local minimum within current interval
        elseif phi_pp > 0 && delta_t < l_interval
            s .+= delta_t * d
            tc = prev_tb + delta_t
            found = true
            pred += phi_p * delta_t + 0.5 * phi_pp * delta_t^2 # Predicted reduction at local min

        # No local minimum in [prev_tb, tb)
        # Prepare for next interval
        else
            # Increment accumulated step and predicted reduction
            s .+= d .* l_interval
            pred += phi_p * l_interval + 0.5 * phi_pp * l_interval^2
            tc = tb

            # Form next search direction
            set_active!(proj_op, idx)
            mul!(d, proj_op, g, -one(T), zero(T)) # d ← P[-g]
            gd = dot(g, d)
            mul!(Hd, hess_op, d)

            # Find next breakpoint
            prev_tb = tb
            gap, idx = next_breakpoint(d, s, slow, supp, proj_op)
            tb = prev_tb + gap
            i += 1
        end
    end

    # Remove zero directions from fixed variables
    set_free!(proj_op, zero_dir)

    return pred
end

"""
    initial_fixed(x, d, xlow, xupp; epsrel = sqrt(eps(T)))

Find the variables that are fixed at the start of the Cauchy step computation.

These are either:

- variables lying at their lower (resp. upper) bound whose component in the search
  direction `d` points out of the feasible region, i.e. is negative (resp. positive)
- variables lying strictly between their bounds whose component in `d` is zero.

# Arguments

- `x`: current iterate
- `d`: search direction
- `xlow`: lower bounds on the variables `x`
- `xupp`: upper bounds on the variables `x`

# Keywords

- `epsrel`: relative tolerance used to decide whether a bound is active and whether a
  component of `d` is zero. Defaults to the square root of the machine precision.

# On return

- `active`: `Vector{Int}` of indices of the variables at an active bound
- `zero_dir`: `Vector{Int}` of indices of the free variables with a zero direction
"""
function initial_fixed(
    x::AbstractVector{T},
    d::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T};
    epsrel::T = sqrt(eps(T))) where T

    active = Vector{Int}()
    zero_dir = Vector{Int}()

    # Components at bounds wiht direction moving out of the feasible region
    for i in axes(x,1)

        # Variable at lower bound with negative direction
        if isfinite(xlow[i]) && x[i] <= xlow[i] + abs(xlow[i])*epsrel && d[i] < epsrel
            push!(active, i)

        # Variable at upper bound with positive direction
        elseif isfinite(xupp[i]) && x[i] + abs(xupp[i])*epsrel >= xupp[i] && d[i] > -epsrel
            push!(active, i)

        # Variable between its bounds but with zero direction
        elseif abs(d[i]) <= epsrel
            push!(zero_dir, i)
        end
    end

    return active, zero_dir
end

"""
    next_breakpoint(d, s, slow, supp, proj_op; epsbp = 10*eps(T))

Find the next breakpoint on the projected gradient path, given the set of variables
already fixed in `proj_op`.

A breakpoint is a step length along `d`, starting from the current step `s`, at which a
free variable reaches one of its bounds `slow` or `supp`. Only the variables that are not
fixed in `proj_op` are considered. Breakpoints that differ by less than `epsbp` are
treated as the same breakpoint, so that several variables can become active at once.

# Arguments

- `d`: current search direction
- `s`: current step on the projected gradient path
- `slow`: lower bounds on the step `s`
- `supp`: upper bounds on the step `s`
- `proj_op`: `Projector` operator that tracks the fixed variables

# Keywords

- `epsbp`: tolerance under which two breakpoints are considered equal. Defaults to
  `10*eps(T)`.

# On return

- `bp_value`: gap between the previous and the next breakpoint, i.e. the step length along
  `d` from `s` to the next breakpoint (`Inf` if no free variable can reach a bound)
- `bp_idx`: `Vector{Int}` of indices of the variables becoming active at that breakpoint
"""
function next_breakpoint(
    d::AbstractVector{T},
    s::AbstractVector{T},
    slow::AbstractVector{T},
    supp::AbstractVector{T},
    proj_op::Projector{T};
    epsbp::T = 10*eps(T)) where T

    bp_value = T(Inf)         # current breakpoint value
    bp_idx = Vector{Int}() # indices of variables becoming active at breakpoint

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
