#=
    cauchy.jl

Computation of the Cauchy step along the projected gradient path, which provides the
starting point for the approximate solution of the trust region subproblem.

Author(s): Pierre Borie
=#

"""
    cauchy_step!(x,s,g,ℓ,u,sₗ,sᵤ,hess_op,proj_op,d,Hd; kwargs...)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model
`q(s) = ½<s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t) = P_D(-t*g)`, for `t ≥ 0`, is the
projected gradient step with `P_D` denoting the projection onto the set of feasible steps
`D = {s | As = 0 and sₗ ≤ s ≤ sᵤ}`, with `sₗ = max(-Δ, ℓ - x)` and
`sᵤ = min(Δ, u - x)`. Equivalently, `x + s(t)` is the projection of `x - t*g` onto the
intersection of the feasible set with the trust region.

This method finds the first local minimum of the quadratic model along the
projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.
The path is continuous and piecewise linear, and the model restricted to each of its pieces
is a one-dimensional quadratic, so that the pieces are examined successively.

The associated Cauchy step is computed in place into vector `s`.

- With a [`CoordinateSubspaceProjector`](@ref) (bounds only), no bound is ever released along
  the path, which follows the procedure of algorithm 17.3.1 from Trust Regions Methods (Conn,
  Gould and Toint, SIAM, 2000).
- With a [`SubspaceProjector`](@ref) (linear equalities and bounds), the bounds are added to
  and released from the working set as the path reaches them and as their multipliers vanish,
  see the documentation of that method.

# Arguments

- `x`: current iterate
- `s`: buffer vector to store the Cauchy step
- `g`: gradient of the augmented Lagrangian at current point
- `ℓ`: lower bounds on the variables `x`
- `u`: upper bounds on the variables `x`
- `sₗ`: lower bounds on the step `s`
- `sᵤ`: upper bounds on the step `s`
- `hess_op`: Operator for the Hessian approximation of type `ALHessian` at current point
- `proj_op`: `Projector` operator onto tangent space of feasible directions
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
    proj_op::CoordinateSubspaceProjector{T},
    d::AbstractVector{T},
    Hd::AbstractVector{T}) where T

    # Constants
    zeroT = T(0.0)
    eps_slope = 1e-10
    eps_curv = 1e-10

    # Fixed variables for the Cauchy step computation and initial projected steepest
    # direction d ← P[-g]
    zero_dir = initial_fixed!(proj_op, x, g, d, xlow, xupp)

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

        # Stop if the projected direction vanishes at machine precision or, after a
        # breakpoint, is so small compared with the gradient that the rounding errors of the
        # projection dominate it; the decrease guarantee only involves the first segment
        if norm(d, Inf) <= (i == 1 ? 10 * eps(T) : sqrt(eps(T))) * norm(g, Inf)
            found = true

        # Stop if slope nonnegative or if slope is small and curvature is positive
        elseif phi_p >= zeroT || (abs(phi_p) <= eps_slope && phi_pp > eps_curv)
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
    cauchy_step!(x,s,g,ℓ,u,sₗ,sᵤ,hess_op,proj_op::SubspaceProjector,d,Hd; kwargs...)

Cauchy step along the exact projected gradient path `s(t) = P_D(-tg)` over the polyhedron
`D = {s | As = 0, sₗ ≤ s ≤ sᵤ}`, the trust region bounds being ordinary bounds of `D`.

# Pieces of the path

On a piece of the path, the working set `𝒲` gathers the bounds held at their value and the
path moves along `d = -P_T g`, where `T = {v | Av = 0, vᵢ = 0 for i ∈ 𝒲}`. The multipliers
`η(t)` of the bounds of `𝒲` in the projection `P_D(-tg)`, i.e. in
`tg + s(t) = Aᵀξ(t) + Σᵢ ηᵢ(t)eᵢ`, are affine in `t`, with slopes `θ` given by the bound
multipliers of the least-squares decomposition `g = P_T g + Aᵀy₁ + E_𝒲θ`. They have the right
sign, i.e. `ηᵢ ≥ 0` for a lower bound and `ηᵢ ≤ 0` for an upper bound. A piece ends at the
first of two events:

- a free variable reaches one of its bounds (hit, see [`next_breakpoint`](@ref)), which is
  then added to the working set;
- the multiplier of a bound of the working set vanishes (release, see
  [`next_release`](@ref)), the bound then leaving the working set unless the direction along
  it points outwards.

At `t = 0` and at each breakpoint, the direction of the next piece is the projection of `-g`
onto the critical cone of the projection, a small quadratic program solved by an active-set
method over the weakly active bounds, i.e. those whose multiplier vanishes (see
[`local_cone_projection!`](@ref)). The multipliers `η` are not unique when the active bounds
and the rows of `A` are linearly dependent; the redundant bounds, which are not represented
in the factor of `proj_op`, keep a zero multiplier.

A piece along which `d` vanishes is stationary: `s` does not move while `η` keeps evolving,
until the next release. Stationary pieces are skipped in the search for the first local
minimizer of `ψ(t) = q(s(t))`. The path ends when `d` vanishes and no release is ahead.

The bounds active at `x` are identified with the same tolerance as in
[`identify_active_set!`](@ref), and the corresponding entries of `sₗ` and `sᵤ` are set to `0`
so that their activity is exact along the path.

# Keywords

- `eta`, `theta`: buffer vectors of size `n` for the multipliers `η` of the active bounds and
  their slopes `θ`, indexed by the variables
- `side`: `Vector{Int8}` of size `n` storing the side of each active bound, `1` for a lower
  bound, `-1` for an upper bound and `0` when both bounds are active (fixed variable)
- `dhat`, `dstep`: buffer vectors of size `n` for [`local_cone_projection!`](@ref)
- `maxpieces`: maximum number of pieces of the path. Defaults to `10(n - m) + 10`. When it is
  reached, the search stops at the current step with a warning.
- `trace`: if a vector is given, the tuple `(t, s(t), d)` is pushed at the start of each
  piece examined, `d` being the direction of the piece (`0` for a stationary piece), and the
  tuple `(t_c, s_c, 0)` is pushed for the Cauchy point. Defaults to `nothing`.

# On return

- `pred`: reduction of the quadratic model obtained after taking the Cauchy step
- argument `s` stores the resulting Cauchy step
- argument `proj_op` encodes the linear equalities and all the bounds active at `x + s`, for
  the conjugate gradient iterations
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
    proj_op::SubspaceProjector{T},
    d::AbstractVector{T},
    Hd::AbstractVector{T};
    eta::AbstractVector{T} = zeros(T, length(x)),
    theta::AbstractVector{T} = zeros(T, length(x)),
    side::AbstractVector{Int8} = zeros(Int8, length(x)),
    dhat::AbstractVector{T} = zeros(T, length(x)),
    dstep::AbstractVector{T} = zeros(T, length(x)),
    maxpieces::Int = 10 * (proj_op.n - proj_op.m) + 10,
    trace::Union{Nothing, AbstractVector} = nothing) where T

    # Constants
    eps_slope = 1e-10
    eps_curv = 1e-10
    epsbp = 10 * eps(T)
    norm_g = norm(g, Inf)
    tol_sign = 10 * eps(T) * norm_g # sign of the multipliers and of the directions
    tol_noise = sqrt(eps(T)) * norm_g # rounding errors of the projection

    # Bounds active at x, with an exact activity along the path
    reset_projector!(proj_op)
    weak = Vector{Int}()
    for i in eachindex(x)
        at_lower = near_lower_bound(x, xlow, i, sqrt(eps(T)), zero(T))
        at_upper = near_upper_bound(x, xupp, i, sqrt(eps(T)), zero(T))
        at_lower && (slow[i] = zero(T))
        at_upper && (supp[i] = zero(T))
        if at_lower || at_upper
            side[i] = at_lower && at_upper ? Int8(0) : (at_lower ? Int8(1) : Int8(-1))
            push!(weak, i)
        end
    end

    # Direction of the first piece, the projection of -g onto the tangent cone at x
    s .= zero(T)
    eta .= zero(T)
    converged = local_cone_projection!(proj_op, g, d, theta, weak, side, dhat, dstep;
                                       tol = tol_sign, tol_step = tol_noise)

    t = zero(T)
    pred = zero(T)
    parked = Vector{Int}()
    piece = 1
    inside_piece = false

    while true

        # The direction is considered to vanish below the rounding errors of the projection,
        # see `cauchy_step!` with a `CoordinateSubspaceProjector`
        tol_dir = (piece == 1 ? 10 * eps(T) : sqrt(eps(T))) * norm_g
        stationary = norm(d, Inf) <= tol_dir
        stationary && fill!(d, zero(T))

        # The free weakly active bounds along which d vanishes stay at their value
        empty!(parked)
        for j in weak
            if !is_fixed(proj_op, j) && side[j] * d[j] <= tol_sign
                d[j] = zero(T)
                push!(parked, j)
            end
        end

        # Next event on the path: a free variable reaches a bound or the multiplier of an
        # active bound vanishes, ties being treated as a single event. A multiplier whose
        # slope is at the level of the rounding errors is not released
        gap_hit, hits = stationary ? (T(Inf), Vector{Int}()) :
            next_breakpoint(d, s, slow, supp, proj_op; epsbp = epsbp)
        gap_rel, rels = next_release(eta, theta, side, proj_op; tol = tol_noise, epsbp = epsbp)
        gap = min(gap_hit, gap_rel)
        gap_hit - gap >= epsbp && empty!(hits)
        gap_rel - gap >= epsbp && empty!(rels)

        trace !== nothing && push!(trace, (t, copy(s), copy(d)))

        # Search for the first local minimizer, skipping stationary pieces
        slope, curv = zero(T), zero(T)
        if !stationary
            mul!(Hd, hess_op, d) # Hd ← H*d
            slope = dot(g, d) + dot(s, Hd)
            curv = dot(d, Hd)

            # Stop if slope nonnegative or if slope is small and curvature is positive
            (slope >= 0 || (abs(slope) <= eps_slope && curv > eps_curv)) && break

            # Positive curvature and local minimum within the current piece
            delta_t = curv > 0 ? -slope / curv : zero(T)
            if curv > 0 && delta_t < gap
                s .+= delta_t .* d
                pred += slope * delta_t + curv * delta_t^2 / 2
                t += delta_t
                inside_piece = true
                break
            end
        end

        # End of the path
        isfinite(gap) || break

        if piece == maxpieces
            @warn "Cauchy step: maximum number of pieces of the projected gradient path " *
                  "($maxpieces) reached"
            break
        end

        # Move to the breakpoint and update the multipliers
        if !stationary
            s .+= gap .* d
            pred += slope * gap + curv * gap^2 / 2

            # A free variable within the rounding errors of the path, of order eps(T)‖tg‖, of
            # a bound it moves towards reaches it at the same breakpoint
            tol_tie = 100 * eps(T) * (t + gap) * norm_g
            for i in eachindex(s)
                (is_fixed(proj_op, i) || i in hits) && continue
                if (d[i] < 0 && s[i] - slow[i] <= tol_tie) ||
                   (d[i] > 0 && supp[i] - s[i] <= tol_tie)
                    push!(hits, i)
                end
            end
        end
        for k in 1:proj_op.p
            i = proj_op.fixidx[k]
            eta[i] += gap * theta[i]
        end
        t += gap

        # Bounds reached by the path, snapped onto their value
        for i in hits
            if d[i] < 0
                s[i], side[i] = slow[i], Int8(1)
            else
                s[i], side[i] = supp[i], Int8(-1)
            end
            eta[i] = zero(T)
        end
        for i in rels
            eta[i] = zero(T)
        end

        # Weakly active bounds: reached, released, parked, and active with a vanishing
        # multiplier, including the redundant ones
        norm_c = zero(T)
        for i in eachindex(s)
            norm_c = max(norm_c, abs(t * g[i] + s[i]))
        end
        tol_eta = 100 * eps(T) * norm_c

        empty!(weak)
        append!(weak, hits)
        append!(weak, parked)
        for i in eachindex(eta)
            if is_fixed(proj_op, i) && abs(eta[i]) <= tol_eta
                eta[i] = zero(T)
                push!(weak, i)
            end
        end

        # Direction of the next piece
        converged &= local_cone_projection!(proj_op, g, d, theta, weak, side, dhat, dstep;
                                            tol = tol_sign, tol_step = tol_noise)
        piece += 1
    end

    converged || @warn "Cauchy step: the active-set method of the local cone projection " *
                       "did not converge"

    trace !== nothing && push!(trace, (t, copy(s), zero(d)))

    # Fix all the bounds active at the Cauchy point for the conjugate gradient iterations: the
    # weakly active bounds if it is a breakpoint, those that stay at their value otherwise,
    # and the bounds reached by coincidence
    set_active!(proj_op, inside_piece ? parked : weak)
    update_inner_active_set!(s, slow, supp, proj_op)

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
    initial_fixed!(proj_op::CoordinateSubspaceProjector, x, g, d, xlow, xupp)

Fix in `proj_op` the variables that are fixed at the start of the Cauchy step computation
on a box and form the corresponding projected steepest direction `d ← P[-g]`.

A variable at a bound is fixed if `-g` points out of the feasible region, which is exactly
the sign condition on its multiplier, see [`initial_fixed`](@ref). The variables lying
strictly between their bounds whose component in the direction is zero are also fixed. They
are freed at the end of the Cauchy step computation.

With a [`SubspaceProjector`](@ref), the working set at the start of the path is computed by
[`local_cone_projection!`](@ref) instead.

# Arguments

- `proj_op`: [`CoordinateSubspaceProjector`](@ref), with no active bound on entry
- `x`: current iterate
- `g`: gradient of the augmented Lagrangian at current point
- `d`: buffer vector to store the projected steepest direction
- `xlow`: lower bounds on the variables `x`
- `xupp`: upper bounds on the variables `x`

# Keywords

- `epsrel`: relative tolerance used to decide whether a component of `d` is zero.
  Defaults to the square root of the machine precision.

# On return

- `zero_dir`: `Vector{Int}` of indices of the free variables with a zero direction
- argument `d` stores the projected steepest direction
- argument `proj_op` encodes the variables fixed at the start of the Cauchy path
"""
function initial_fixed!(
    proj_op::CoordinateSubspaceProjector{T},
    x::AbstractVector{T},
    g::AbstractVector{T},
    d::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T};
    epsrel::T = sqrt(eps(T))) where T

    # Initial projected steepest direction
    mul!(d, proj_op, g, -one(T), zero(T)) # d ← P[-g]

    # Fixed variables for the Cauchy step computation
    active, zero_dir = initial_fixed(x, d, xlow, xupp; epsrel = epsrel)
    fixed = vcat(active, zero_dir)

    # Update the projector operator and search direction
    !isempty(fixed) && set_active!(proj_op, fixed)
    mul!(d, proj_op, g, -one(T), zero(T)) # d ← P[-g]

    return zero_dir
end

"""
    projected_direction!(d, θ, P::SubspaceProjector, g)

Form the projected steepest direction `d ← P[-g]` and copy into the vector `θ` of size `n`
the multipliers of the active bounds in the decomposition `g = Pg + Aᵀy₁ + E_𝒜θ` (see
[`copy_bound_multipliers!`](@ref)).
"""
function projected_direction!(
    d::AbstractVector{T},
    theta::AbstractVector{T},
    P::SubspaceProjector{T},
    g::AbstractVector{T}) where T

    mul!(d, P, g, -one(T), zero(T)) # d ← P[-g]
    copy_bound_multipliers!(theta, P)
    return d
end

"""
    local_cone_projection!(proj_op, g, d, θ, weak, side, dhat, dstep; tol, maxit)

Compute the direction of the projected gradient path at a point of the path, i.e. at `t = 0`
or at a breakpoint, by projecting `-g` onto the critical cone of the projection.

The bounds active at the current point are either strongly active, those fixed in `proj_op`
with a nonzero multiplier, or weakly active, those of `weak`. The direction solves

`min_d ½‖d + g‖²   s.t.   Ad = 0,   dᵢ = 0 for the strongly active bounds,`

`                         σᵢdᵢ ≥ 0 for i ∈ weak,`

where `σᵢ = side[i]` is `1` for a lower bound and `-1` for an upper bound. The variables of
`weak` with both bounds active (`side[i] = 0`) stay fixed.

The problem is solved by a primal active-set method (Nocedal and Wright, Numerical
Optimization, algorithm 16.3) whose working set is encoded in `proj_op`. The method starts
with all the weakly active bounds fixed, so that the first iterate is feasible. A weakly
active bound whose multiplier has the wrong sign (see [`wrong_sign_bounds`](@ref)) leaves the
working set, and a free weakly active bound along which the step points outwards blocks it
and enters the working set. After a step of zero length, the smallest index rule chooses
the leaving bound. At a breakpoint where a single bound is reached or released, the method
ends after zero or one change of the working set.

When the active bounds and the rows of `A` are linearly dependent, freeing a bound may only
bring a redundant bound into the factor of `proj_op`, and fixing a redundant bound leaves the
factor unchanged. The subspace, of unchanged dimension, is then the same, and its minimizer
is kept exactly, so that the rounding errors of the projection cannot make the method cycle.
For the same reason, the components of the steps below `tol_step` are ignored in the ratio
test.

# Arguments

- `proj_op`: [`SubspaceProjector`](@ref), in which the strongly active bounds are fixed
- `g`: gradient of the model at `s = 0`
- `d`: buffer vector to store the direction
- `θ`: buffer vector of size `n` to store the multipliers
- `weak`: `Vector{Int}` of the indices of the weakly active bounds
- `side`: `Vector{Int8}` of the sides of the active bounds
- `dhat`, `dstep`: buffer vectors

# Keywords

- `tol`: tolerance on the sign of the multipliers. Defaults to `10 eps(T) ‖g‖∞`.
- `tol_step`: tolerance on the components of the steps, below which a step is considered
  zero and a component does not block it. Defaults to `√eps(T) ‖g‖∞`.
- `maxit`: maximum number of iterations. Defaults to `10 length(weak) + 10`.

# On return

- `true` if the method has converged, `false` if the maximum number of iterations has been
  reached, in which case `d` is a feasible descent direction of the problem
- argument `d` stores the direction `-P g`, where `P` projects onto the subspace encoded in
  `proj_op`
- argument `θ` stores the multipliers of the decomposition `g = Pg + Aᵀy₁ + E_𝒲θ` over the
  bounds of the working set represented in the factor of `proj_op`, indexed by the
  variables
- argument `proj_op` encodes the working set of the next piece of the path
"""
function local_cone_projection!(
    proj_op::SubspaceProjector{T},
    g::AbstractVector{T},
    d::AbstractVector{T},
    theta::AbstractVector{T},
    weak::AbstractVector{Int},
    side::AbstractVector{Int8},
    dhat::AbstractVector{T},
    dstep::AbstractVector{T};
    tol::T = 10 * eps(T) * norm(g, Inf),
    tol_step::T = sqrt(eps(T)) * norm(g, Inf),
    maxit::Int = 10 * length(weak) + 10) where T

    # Feasible start, with all the weakly active bounds fixed
    set_active!(proj_op, weak)
    projected_direction!(dhat, theta, proj_op, g)
    d .= dhat
    stalled = false

    for _ in 1:maxit

        dstep .= dhat .- d

        if norm(dstep, Inf) <= tol_step
            # d minimizes the objective on the working set: free the bound whose multiplier
            # has the most negative sign, or the smallest index if the iterate is stalled
            d .= dhat
            j, worst = 0, zero(T)
            for i in weak
                (side[i] == 0 || !is_represented(proj_op, i)) && continue
                viol = side[i] * theta[i]
                if viol < -tol && (j == 0 || (stalled ? i < j : viol < worst))
                    j, worst = i, viol
                end
            end

            # Multipliers of the right sign: the direction is found
            j == 0 && return true

            dof = nb_degrees_of_freedom(proj_op)
            set_free!(proj_op, j)
            stalled = true
            projected_direction!(dhat, theta, proj_op, g)

            # A redundant bound has replaced j in the factor: the subspace is unchanged, and
            # so is its minimizer, only the multipliers change
            nb_degrees_of_freedom(proj_op) == dof && (dhat .= d)
        else
            # Longest step towards the minimizer on the working set that keeps the free
            # weakly active bounds satisfied, the smallest index blocking on ties
            alpha, block = one(T), 0
            for i in weak
                is_fixed(proj_op, i) && continue
                rate = side[i] * dstep[i]
                rate < -tol_step || continue
                ratio = max(side[i] * d[i], zero(T)) / (-rate)
                if ratio < alpha || (block > 0 && ratio == alpha && i < block)
                    alpha, block = ratio, i
                end
            end
            alpha > 0 && (stalled = false)

            # Full step: the working set is unchanged
            if block == 0
                d .= dhat
                continue
            end

            axpy!(alpha, dstep, d) # d ← d + α(d̂ - d)
            dof = nb_degrees_of_freedom(proj_op)
            set_active!(proj_op, block)

            # A redundant blocking bound leaves the subspace and the multipliers unchanged
            nb_degrees_of_freedom(proj_op) == dof ||
                projected_direction!(dhat, theta, proj_op, g)
        end
    end

    return false
end

"""
    next_release(η, θ, side, proj_op; tol = 0, epsbp = 10*eps(T))

Find the next release on the projected gradient path over a polyhedron, i.e. the step length
along the current piece at which the multiplier of an active bound vanishes.

On a piece of the path, the multipliers of the bounds of the working set encoded in
`proj_op` are `η + τθ`, `τ ≥ 0` being the step length from the start of the piece. The
multiplier of bound `i`, of side `σᵢ = side[i]`, has the right sign as long as
`σᵢ(ηᵢ + τθᵢ) ≥ 0`, so that it vanishes at `τ = σᵢηᵢ / (-σᵢθᵢ)` if `σᵢθᵢ < -tol`. Only the
bounds represented in the factor of `proj_op` are considered, and the variables whose two
bounds are active (`σᵢ = 0`) are never released. Releases that differ by less than `epsbp`
are treated as the same release.

# Arguments

- `η`: multipliers of the active bounds at the start of the piece, indexed by the variables
- `θ`: slopes of these multipliers on the piece (see [`copy_bound_multipliers!`](@ref))
- `side`: `Vector{Int8}` of the sides of the active bounds
- `proj_op`: [`SubspaceProjector`](@ref) encoding the working set of the piece

# Keywords

- `tol`: nonnegative tolerance on the sign of `σᵢθᵢ`. Defaults to `0`.
- `epsbp`: tolerance under which two releases are considered equal. Defaults to
  `10*eps(T)`.

# On return

- `rel_value`: step length from the start of the piece to the next release (`Inf` if no
  multiplier vanishes)
- `rel_idx`: `Vector{Int}` of indices of the variables whose bound is released
"""
function next_release(
    eta::AbstractVector{T},
    theta::AbstractVector{T},
    side::AbstractVector{Int8},
    proj_op::SubspaceProjector{T};
    tol::T = zero(T),
    epsbp::T = 10*eps(T)) where T

    rel_value = T(Inf)
    rel_idx = Vector{Int}()

    for k in 1:proj_op.p
        i = proj_op.fixidx[k]
        side[i] == 0 && continue

        rate = side[i] * theta[i]
        rate < -tol || continue

        # A multiplier of the wrong sign by rounding is released at once
        rel_try = max(side[i] * eta[i], zero(T)) / (-rate)

        if abs(rel_value - rel_try) < epsbp
            push!(rel_idx, i)
        elseif rel_try < rel_value
            rel_value = rel_try
            empty!(rel_idx)
            push!(rel_idx, i)
        end
    end

    return rel_value, rel_idx
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
