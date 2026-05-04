export traulls

"""
    solve(model; kwargs...)

Solve a bound-constrained nonlinear least-squares problem with equality
constraints of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0`

`Ax = b`

`ℓ ≤ x ≤ u,`

with an iterative Augmented Lagrangian method.

Starting from an initial guess `x₀` and an initial estimate of the vector of
Lagrange multipliers associated to the equality constraints `y₀`,
each new iterate `xₖ₊₁` is an approximate solution, with respect to a tolerance
`ωₖ > 0`, of the subproblem

`minₓ Lₐ(x,yₖ,μₖ) = 1/2 * r(xₖ)ᵀr(xₖ) + c(xₖ)ᵀ[yₖ + μₖ/2 * c(xₖ)]`

`s.t. Ax = b`

`ℓ ≤ x ≤ u,`

using `xₖ` as a starting point with fixed penalty parameter `μₖ > 0` and current estimate of
the Lagrange multipliers `yₖ`

If the new iterate satisfies `||c(xₖ₊₁)||₂ ≤ ηₖ`, for some `ηₖ > 0`, then the
Lagrange multipliers are updated by `yₖ₊₁ = yₖ + μₖc(xₖ)` and the tolerances `ωₖ` and `ηₖ` are tightened.

On the contrary, if xₖ₊₁ fails to satisfies the feasibility inequality, the
iterate is unchanged, i.e. `(xₖ₊₁,yₖ₊₁) = (xₖ,yₖ)` and the minimization of the
subproblem is restarted with
a higher penalty parameter `μₖ₊₁ = τμₖ`, with `τ > 1`. The tolerances `ωₖ` and
`ηₖ` are still reduced but in a weaker maner.

Subproblems are solved by the gradient projection method
see [`projected_gradient`](@ref)).

# Arguments

- `model::CnlsModel`: Encodes the model of the problem to be solved (see [`CnlsModel`](@ref)).

# Keyword Arguments

## Augmented Lagrangian parameters

- `mu`: Initial penalty parameter (default: `10`)
- `tau`: Increase factor for the penalty parameter (default: `10`)
- `omega0`: Constant to set the initial criticality tolerance
(default: `1`)
- `eta0`: Constant to set the initial feasibility tolerance
(default: `1`)
- `min_tol_feas`: Absolute tolerance for feasibility of nonlinear constraints
(default: `1e-6`)
- `crit_tol`: Relative tolerance for criticality (default: `1e-7`)
- `k_crit`: Positive constant used to initialize and update the
subproblem criticality tolerance in the case of poor improvement of the
feasibility (default: `1.0`)
- `k_feas`: Positive constant used to initialize and update the subproblem feasibility
tolerance in the case of poor improvement of the feasibility (default: `0.1`)
- `beta_crit`: Positive constant used to reduce the subproblem criticality tolerance in the
case of good improvement of the primal feasibility (default: `1.0`)
- `beta_feas`: Positive constant used to reduce the subproblem feasibility tolerance in the
case of good improvement of the feasibility (default: `0.9`)
- `hessian_approx`: Symbol encoding the Hessian approximation used during the inner
minimization (default: `:gn` for Gauss-Newton)

## Trust region parameters

- `accept_treshold`: Threshold for accepting a step (default: `0.05`)
- `increase_treshold`: Threshold for very successful steps to extend the trust region
(default: `0.9`)
- `decrease_factor`: Reducing factor of the trust region (default: `0.25`)
- `increase_factor`: Extension factor of the trust region (default: `2.5`)
- `neg_ratio_factor`: Reduction factor of the trust region radius for steps with negative
ratio (default: `0.0625`)

## Solver related constants

- `mu_max`: maximum value of the penalty parameter (default: `1/ϵ` with `ϵ` is the relative
machine precision)
- `max_outer_iter`: Maximum number of outer iterations (default: `200`)
- `max_inner_iter`: Maximum number of iterations for the inner minimization phase
(default: `1000`)
- `max_cg_iter`: Maximum number of minor iterates for the gradient projection loop
(default: `50`)

## Miscellaneous

- `output_io`: Input-Output stream where the iteration detail is printed (default: `stdout`)
- `verbose`: Boolean. If set to `true`, print information about the outer iterations into
`output_io` (default: `false`)
- `inner_verbose`: Boolean. If set to `true`, print information about the iterations of the
inner minimization phase into `output_io` (default: `false`)

# On return

Returns the important execution informations into a [`TraullsResults`](@ref).
"""
function traulls(
    model::CnlsModel{T};
    mu::T = T(10),
    tau::T = T(10),
    omega0::T = T(1),
    eta0::T = T(1),
    min_tol_feas::T = T(1e-7),
    min_reltol_crit::T = T(1e-7),
    k_crit::T = T(1),
    k_feas::T = T(1//10),
    beta_crit::T = T(1),
    beta_feas::T = T(9//10),
    accept_threshold::T = T(0.05),
    increase_threshold::T = T(0.9),
    decrease_factor::T = T(0.25),
    increase_factor::T = T(2.5),
    neg_ratio_factor::T = T(0.0625),
    hessian_approx::Symbol = :gn,
    mu_max::T = 1/eps(T),
    max_iter::Int = 200,
    max_inner_iter::Int = 1000,
    max_cg_iter::Int = 50,
    output_io::IO=stdout,
    verbose::Bool=false,
    inner_verbose::Bool=false) where T

    # Arguments sanity checks
    !(hessian_approx in keys(dict_hessians)) &&
        throw(ArgumentError("Wrong hessian argument has been passed. Supported keywords are " *
        string(keys(dict_hessians))))

    # Dimensions of the problem
    n, nres, ncons = model.n, model.nres, model.ncons
    lincons_present = model.nlincons > 0

    # Make initial point feasible and form subspace projector operator
    x = model.x
    proj_op = initial_point_and_projector!(model, x, Val(lincons_present))
    xlow, xupp = model.xlow, model.xupp

     # Allocate memory for buffer vectors involved in inner minimization
    inner_workspace = Workspace(T, n, nres ,ncons)

    # Allocate buffers for functions and first derivatives evaluation
    rx = residuals(model, x)
    cx = nlconstraints(model, x)
    J = jac_residuals(model, x)
    C = jac_nlconstraints(model, x)

    y = least_squares_multipliers(rx, J, C) # Initial Lagrange mutipliers
                                            # estimates
    g = al_grad(rx, cx, y, mu, J, C)        # Gradient of the AL
    model.counters.nalgrad_eval += 1

    # Ininitialize Hessian approximation
    hessian_type = dict_hessians[hessian_approx]
    hess_op = @match hessian_type begin
        $gn     => GN(J, C, mu)
        $sr1    => SR1(J, C, mu)
        $hybrid_bfgs => HybridBFGS(J, C, mu, rx, cx, y)
    end

    # Set up trust region
    tr = TrustRegion(accept_threshold,
                     increase_threshold,
                     decrease_factor,
                     increase_factor,
                     neg_ratio_factor)

    # Set up tolerances
    reltol_crit, tol_feas = initial_tolerances(mu, omega0, eta0, k_crit, k_feas)

    # Initial values of objective, feasibility and criticality
    g = J'*rx + C'*y
    fx = dot(rx, rx)

    model.counters.nobj_eval += 1
    feas_measure = norm(cx, Inf)
    gproj = inner_workspace.proj_g

    pix = lincons_present ?
        criticality_measure(x, g, gproj, proj_op) :
        criticality_measure(x, g, gproj, xlow, xupp)

    # tol_scale_factor = max(1, norm(g, Inf))
    tol_crit = min_reltol_crit * (1 + pix)

    solved = feas_measure <= min_tol_feas && pix <= tol_crit

    stopped, max_penalty_reached = false, false

    iter = 1

    verbose && print_traulls_header(model, fx, feas_measure, pix, min_reltol_crit,
                                    min_tol_feas, tau, tr; io=output_io)

    # Set counters
    reset!(model.counters)
    start_time = time()

    while !solved && iter <= max_iter && !max_penalty_reached

        pix = solve_subproblem!(
            model,
            x,
            xlow,
            xupp,
            y,
            mu,
            rx,
            cx,
            J,
            C,
            g,
            hess_op,
            proj_op,
            tr,
            reltol_crit,
            hessian_type,
            max_inner_iter,
            max_cg_iter,
            inner_workspace;
            verbose=inner_verbose,
            io=output_io)


        # Evaluate feasibility and objective
        feas_measure = norm(cx, Inf)
        g .= J'*rx + C'*y
        fx  = dot(rx,rx)

        update_multipliers = feas_measure <= tol_feas

        if update_multipliers

            # Update Lagrange multipliers
            first_order_multipliers!(y, cx, mu)

            # Decrease tolerances
            # Penalty parameter is unchanged
            reltol_crit = max(reltol_crit / mu^beta_crit, min_reltol_crit)
            tol_feas = max(tol_feas / mu^beta_feas, min_tol_feas)

        else
            # Increase the penalty parameter lesser decrease of the tolerances
            # Iterate and multipliers are unchanged
            mu *= tau
            reltol_crit = max(omega0 / mu^k_crit, min_reltol_crit)
            tol_feas = max(eta0 / mu^k_feas, min_tol_feas)
        end


        # Evaluate termination status
        g .= J'*rx + C'*y # Lagrangian gradient

        norm_proj_gradlag = lincons_present ?
            criticality_measure(x, g, gproj, proj_op) :
            criticality_measure(x, g, gproj, xlow, xupp)

        solved = feas_measure <= min_tol_feas &&
            norm_proj_gradlag <= tol_crit


        verbose && print_outer_iteration(
            iter,
            fx,
            feas_measure,
            mu,
            pix,
            Val(update_multipliers);
            io=output_io)

        max_penalty_reached = mu >= mu_max

        iter += 1

    end

    solving_status = if solved
        first_order_critical
        elseif feas_measure <= min_tol_feas
        feasible_non_critical
        elseif max_penalty_reached
        penalty_too_high
        else
        infeasible_non_critical
    end

    elapsed_time = time() - start_time     # Save execution time
    model.counters.niter_outer = iter - 1  # Save number of outer iterations

    results = TraullsResults(x, y, fx, feas_measure, pix, solving_status, model.counters,
                             elapsed_time)

    verbose && print(output_io, results)

    return results

end

"""
    solve_subproblem!(model, args...)

Solves the outer iteration subproblem

`minₓ Lₐ(x,y,μ) = 1/2 * r(x)ᵀr(x) + c(x)ᵀ[y + μ/2 * c(x)]`

`s.t. Ax = b`

`ℓ ≤ x ≤ u,`

using the gradient projection method with trust region.

The starting point `x₀` and optimality tolerance `ω` are given. The Lagrange
multipliers `y` and penalty parameter `μ` are fixed.

At iteration `k`, a quadratic model of the objective function around `xₖ` is
formed by

`qₖ(s) = 1/2 sᵀHₖs + sᵀgₖ,`

with `gₖ = ∇ₓLₐ(xₖ,y,μ)` and `Hₖ ≈ ∇²ₓₓ Lₐ(xₖ,y,μ)`.

The step computation consists into approximately solving the quadratic program

`minₛ qₖ(s)`

`s.t. A(x + s) = b`

`ℓ ≤ xₖ + s ≤ u`

`||s|| ≤ Δₖ,`

where `Δₖ` is the trust region radius and `||.||` denotes the `∞`-norm
`||x|| = maxᵢ |xᵢ|`. Because `||x|| ≤ Δₖ ⟺ -Δₖ ≤ xᵢ ≤ Δₖ` for all `i`,
the feasible domain for the step can actually be formulated as the box

`Bₖ = [max(-Δₖe, ℓ-x), min(Δₖe, u-x)]`, with `e = (1,...,1)`.

# Solving the QP

## Cauchy point

We start by finding the first local minimizer of the model along the projected
gradient path

`s(t) = Pₖ[xₖ - tgₖ] - xₖ` for  `t ≥ 0,`

`Pₖ` denoting the projection over the feasible domain.
The corresponding scalar defines a Cauchy step that ensures a sufficient
reduction of the objective function. This means that taking the Cauchy step at
every iteration is enough to solve the subproblem.

## Beyond the Cauchy point

In order to provide a better reduction, we then apply the conjugate gradient
method to the subspace where the components corresponding to bounds active at
the Cauchy point are fixed.

The resulting `sₖ` step is then accepted or rejected depending on the value of
 the ratio of the actual reduction over the reduction predicted by the model

`ρ = (Lₐ(xₖ+sₖ,y,μ) - Lₐ(xₖ,y,μ)) / qₖ(sₖ) - qₖ(0)`.

If `ρ ≥ η₁`, where `η₁ ∈ (0,1)` is a given parameter, then the step is accepted
and the radius `Δₖ` is eventually increased.
This translates the fact that there is a good agreement between the objective
function and the model.

If `ρ < η₁` (poor agreement), the step is rejected and the minimization is
restarted with a smaller trust region.

## Trust region update

The scalars `η₁, η₂, α₁, α₂, γᵦ` are constant chosen such that

`0 < η₁ ≤ η₂ < 1`, `0 < α₁ < 1 < α₂` and `0 < γᵦ < 1`.

The radius is updated as follows:
- if `ρ ≥ η₂` (very good step), `Δₖ₊₁ = max(α₂*||sₖ||, Δₖ)`
- if `η₁ ≤ ρ < η₂` (good step), `Δₖ₊₁ = Δₖ`
- if `0 < ρ < η₁` (bad step), `Δₖ₊₁ = α₁*||sₖ||`
- if `ρ ≤ 0` (very bad step), `Δₖ₊₁ = min(α₂*||sₖ||, γᵦ*Δₖ)`

Here, `||.||` denotes the ∞-norm.

## Stopping criteria

The minimization process is stopped once there is an iterate `xₖ` such that

`|| P[xₖ - gₖ] - xₖ || ≤ ω * || P[x₀ - g₀] - x₀ ||`,

where `P` here denotes the projection operator onto the tangent space of feasible directions
 at `x`, i.e. the norm of the reduced gradient.

The algorithm also stops if there are consecutive iterations provide a relatively small
change in both the objective and the iterate or if the trust region radius is too small.
# Arguments

- `model`: `CnlsModel` encoding the original problem to be solved
- `x`: Starting point the the outer iteration
- `xlow`: Lower bounds on the variables
- `xupp`: Upper bounds on the variables
- `y`: Current estimation of the Lagrange multipliers
- `mu`: Penalty parameter
- `rx`: Residuals evaluated at `x`
- `cx`: Nonlinear constraints evaluated at `x`
- `J`: Jacobian of the residuals evaluated at `x`
- `C`: Jacobian of the nonlinear constraints evaluated at `x`
- `g`: Gradient of the augmented Lagrangian at `x`
- `hess_op`: `ALHessian` operator to compute Hessian-vector products
- `proj_op`: `Projector` operator to compute projections onto tangent spaces
- `tr`: `TrustRegion` encoding the trust region constraint and the constants involved in the
radius update mechanism (see [`TrustRegion`](@ref))
- `rel_tol_crit`: Relative optimality tolerance
- `hessian_approx`: Instance of `HessianApprox` enum type encoding how the approximated
Hessian is updated
- `max_iter::Int`: maximum number of inner iterations
- `max_cg_iter::Int`: maximum number of uses of the conjugate gradient method, i.e. minor
iterates
- `workspace`: `Workspace` whose fields contain pre-allocated memory for vectors involed in
linear algebra computations
- `patience_counter`: Maximum number of consecutive stalling iterations before exiting the
algorithm (default: `3`)
- `verbose`:Boolean. If set to `true`, print iteration detail into `io` (default: false)
- `io`: input/output stream to log iteration detail (default: `stdout`)
"""
function solve_subproblem!(
    model::CnlsModel{T},
    x::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T},
    y::AbstractVector{T},
    mu::T,
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    g::AbstractVector{T},
    hess_op::ALHessian{T},
    proj_op::Projector{T},
    tr::TrustRegion{T},
    reltol_crit::T,
    hessian_approx::HessianApprox,
    max_iter::Int,
    max_cg_iter::Int,
    workspace::Workspace{T};
    patience_counter::Int = 3,
    verbose::Bool=false,
    io::IO=stdout) where T

    # Dimensions
    n, nslack, ncons = model.n, model.nslack, model.ncons
    lincons_present = model.nlincons > 0

    # Buffers to save previous iterate and functions evaluations
    x_prev = workspace.x_prev
    rx_prev = workspace.rx_prev
    cx_prev = workspace.cx_prev

    s = workspace.step       # step
    gproj = workspace.proj_g # projected gradient

    # Evaluate objective and gradient of the AL at current point (x,y)
    alx = al_objgrad!(rx, cx, y, mu, J, C, g)
    model.counters.nalgrad_eval += 1
    model.counters.nalobj_eval += 1
    # Reset Hessian approximation and projector operator
    @match hessian_approx begin
        $gn     => reset_hessian!(hess_op, J, C, mu)
        $sr1    => reset_hessian!(hess_op, J, C, mu)
        $hybrid_bfgs => reset_hessian!(hess_op, J, C, mu, rx, cx, y)
    end

    reset_projector!(proj_op)

    # Initialize trust region
    set_initial_radius!(tr, g)

    # Prepare for inner minimization loop
    # TODO: add computation of criticality measure for polyhedral problem
    # (when lincons_present = true)
    pix = lincons_present ?
        criticality_measure(x, g, gproj, proj_op) :
        criticality_measure(x, g, gproj, xlow, xupp)

    tol_crit = reltol_crit * (1 + pix)
    solved = pix <= tol_crit
    short_circuit = false
    plateau_iter = 0
    iter = 1

    verbose && print_inner_iter_header(io)

    while !solved && iter <= max_iter && !short_circuit

        x_prev .= x
        rx_prev .= rx
        cx_prev .= cx
        alx_prev = alx

        radius = tr.radius

        pred = projected_gradient!(
            x,
            s,
            g,
            gproj,
            hess_op,
            proj_op,
            xlow,
            xupp,
            radius,
            max_cg_iter,
            reltol_crit,
            reltol_crit,
            workspace)

        if short_circuit continue end

        # Evaluate the objective at trial point
        x .+= s
        residuals!(model, rx, x)
        nlconstraints!(model, cx, x)
        alx = al_obj(rx, cx, y, mu)
        model.counters.nalobj_eval += 1
        norm_step = norm(s, Inf) # used for radius update

        # Step taken on the slack variables, if any
        if nslack > 0

            # Add "magical" step to current point x
            step_slack!(x, y, cx, mu, nslack, ncons)

            # Adjust the step vector
            slack_idx = n - nslack + 1 : n
            ineq_idx = ncons - nslack + 1 : ncons
            s[slack_idx] .= x[slack_idx] .- x_prev[slack_idx] .- s[slack_idx]

            # Update the constraints involving slack variables without evaluating
            cx[ineq_idx] .-= s[slack_idx]

            # Add reduction of the true objective function after taking second
            # step to pred
            pred -= alx
            alx = al_obj(rx, cx, y, mu)
            model.counters.nalobj_eval += 1
            pred += alx

        end

        # Compute the ratio actual reduction / predicted reduction
        ratio = step_ratio(alx_prev, alx, pred)

        accepted = accept_step(tr, ratio)

        if accepted

            # Evaluate first derivative at trial point

            jac_residuals!(model, J, x)
            jac_nlconstraints!(model, C, x)
            al_grad!(rx, cx, y, mu, J, C, g)
            model.counters.nalgrad_eval += 1

            # Update Hessian approximation
            if hessian_approx == gn
                # Gauss-Newton case
                update_hessian!(hess_op, J, C)

            elseif hessian_approx == sr1
                # SR1 update
                update_hessian!(hess_op, J, C, rx, cx, g, y, s)

            elseif hessian_approx == hybrid_bfgs
                # HybridBFGS update
                update_hessian!(hess_op, J, C, rx, cx, g, y, s)
            end

            pix = lincons_present ?
                criticality_measure(x, g, gproj, proj_op) :
                criticality_measure(x, g, gproj, xlow, xupp)


        else
            x .= x_prev
            rx .= rx_prev
            cx .= cx_prev
            alx = alx_prev
        end

        # norm_step = norm(s, Inf)
        update_radius!(tr, ratio, norm_step)

        verbose && print_inner_iter(iter, alx_prev, norm(s, Inf), radius, ratio;io=io)

        # Evaluate termination criteria
        solved = pix <= tol_crit

        # Stop if stalling on `patience_counter` consecutive accepted steps or trust region
        # has shrinked too much
        stalling = check_stalling(s, x_prev, alx_prev, alx, accepted)
        plateau_iter = stalling ? plateau_iter + 1 : 0
        shrinked_tr = small_radius(x, radius)
        short_circuit = plateau_iter == patience_counter || shrinked_tr

        iter += 1
    end

    # Indicate inner iteration finish
    verbose && println(io,'='^92)
    # Save number of inner iterations
    model.counters.niter_inner += iter

    return pix
end

"""
    projected_gradient!(x,g,z,xₗ,xᵤ,Δ,max_cg_iter,κₛ,κᵪ)

Approximately solves the quadratic program

`minₛ 1/2 sᵀHs + sᵀg`

`s.t. As = 0`

`xₗ ≤ x + s ≤ xᵤ`

`||s|| ≤ Δ`

by the gradient projection method.

In the QP model, `||.||` denotes the `∞`-norm `||s|| = maxᵢ |sᵢ|`.

# Arguments

- `x`: Current iterate
- `s`: Buffer vector for the step
- `g`: Gradient of the Augmented Lagrangian at `x`
- `z`: Buffer vector for the projected gradient
- `H`: `ALHessian` operator to compute Hessian-vector products
- `P`: `Projector` operator to compute projections onto tangent spaces
- `xₗ`: Lower bounds on `x`
- `xᵤ`: Upper bounds on `x`
- `Δ::T`: Trust region radius
- `max_cg_iter::Int`: Number of maximum uses of the conjugate gradient method
- `κₛ::T`: Positive constant used to define the convergence criteria
relative of the gradient projection method
- `κᵪ::T`: Positve constant used to define the convergence criteria of
the conjugate gradient method

# On return

- `s`: This argument is modified in place and contains the trial step
- `pred`: Reduction of the quadratic model after taking step `s`
"""
function projected_gradient!(
    x::AbstractVector{T},
    s::AbstractVector{T},
    g::AbstractVector{T},
    gproj::AbstractVector{T},
    hess_op::ALHessian{T},
    proj_op::Projector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T},
    radius::T,
    max_cg_iter::Int,
    kappa_pg::T,
    kappa_cg::T,
    workspace::Workspace{T}) where T

    # Buffers
    Hs = workspace.hess_vec
    slow, supp = workspace.step_low, workspace.step_upp
    w = workspace.search_dir
    r, v, p = workspace.r, workspace.v, workspace.p

    # Bounds the step  on the search direction
    slow .= (t -> max(-radius, t)).(xlow-x)
    supp .= (t -> min(radius, t)).(xupp-x)
    # Reset active constraints
    reset_projector!(proj_op)

    pred = cauchy_step!(x,
                 s,
                 g,
                 xlow,
                 xupp,
                 slow,
                 supp,
                 hess_op,
                 proj_op,
                 gproj,
                 Hs)

    # Set up for conjugate gradient iterations
    mul!(Hs, hess_op, s)

    b = workspace.cg_rhs
    b .= Hs .+ g

    # If Cauchy step provides sufficient decrease, exit
    quasi_optimal = norm(proj_op * b) <= kappa_pg * norm(proj_op*g)
    cg_stop = false
    iter = 1

    while !quasi_optimal && !cg_stop && iter <= max_cg_iter && !saturated_subspace(proj_op)

        cg_status, pred_cg = pcg!(
            b,
            hess_op,
            proj_op,
            s,
            slow,
            supp,
            radius,
            r,
            v,
            p,
            Hs,
            kappa_cg)

        # Increment predicted reduction
        pred += pred_cg

        # Prepare for next CG iterations
        mul!(Hs, hess_op, s) # form Hs
        b .= Hs .+ g

        # Compute norms of reduced gradients ||Zᵀg|| and ||Zᵀ(Hs+g)||
        # TODO: in-place computations for norms of reduced gradients
        norm_reduced_g = norm(proj_op * g)
        norm_reduced_gnext = norm(proj_op * b)

        # Stop if the step provides sufficient decrease in the reduced gradient
        quasi_optimal = norm_reduced_gnext <= kappa_pg * norm_reduced_g

        # Stop if negative curvature encountered
        cg_stop = cg_status == negative_curvature # || cg_status == on_trust_region

        # Identify the newly active bounds and update accordingly the projection operator
        update_active_set!(s, x, xlow, xupp, proj_op)

        iter += 1
    end

    return pred
end


# Modifies the initial guess for the solution such that it is feasible to the bounds
# Forms and returns the operator computing projections on coordinate subspaces
function initial_point_and_projector!(
    model::CnlsModel{T},
    x::AbstractVector{T},
    ::Val{false}) where T

    # Make starting point feasible with respect to bounds
    x .= max.(model.xlow, min.(x, model.xupp))

    CoordinateSubspaceProjector(model.n; T=T)
end

# Modifies the initial guess for the solution to make it linear feasible
# Identifies the bounds active at the initial point found and forms the associate `Projector
# operator computing projections on reduced subspaces
function initial_point_and_projector!(
    model::CnlsModel{T},
    x::AbstractVector{T},
    ::Val{true};
    tol::T=sqrt(eps(T))) where T

    A, b = model.linmat, model.linrhs
    n, m = model.n, model.nlincons
    xlow, xupp = model.xlow, model.xupp

    solve_linfeas_pb!(A, x, b, xlow, xupp, m, n)

    # Identity active bounds at starting point
    initial_active = falses(n)

    for i in axes(x,1)
        initial_active[i] = x[i] <= xlow[i] + tol || xupp[i] <= x[i] + tol
    end

    chol_aat = Cholesky(A*A')

    # Form projector operator
    SubspaceProjector(A, initial_active, chol_aat)
end

# Solves w.r.t. x and auxiliary variables r the linear feasibility problem
# `min ||r||₁ s.t. Ax + r = b, ℓ ≤ x ≤ u`
# Modifies in place argument x0 with the value at optimal solution
# Triggers a warning if the solution found is not feasible
function solve_linfeas_pb!(
    A::AbstractMatrix{T},
    x0::AbstractVector{T},
    b::AbstractVector{T},
    lb::AbstractVector{T},
    ub::AbstractVector{T},
    m::Int,
    n::Int) where T

    feas_model = Model(HiGHS.Optimizer)
    set_silent(feas_model)

    @variable(feas_model, lb[i] <= x[i=1:n] <= ub[i], start = x0[i])
    @variable(feas_model, r[1:m])

    @constraint(feas_model, A*x + r == b)

    # Model 1-norm of residual variable r
    @variable(feas_model, t)
    @constraint(feas_model, [t; r] in MOI.NormOneCone(m + 1))

    @objective(feas_model, Min, t)

    optimize!(feas_model)

    #TODO: Handle failed linear feasibility to stop the solver
    value(t) > 0 && @warn("Failed to achieve linear feasibility")

    x0 .= value.(x)

    return
end


"""
    criticality_measure(x,g,xₗ,xᵤ)

Computes the criticality measure for problems where linear constraints are bounds.

# Arguments

- `x`: Current iterate
- `g`: Gradient of the Augmented Lagrangian at current primal-dual
iterate
- `gproj`: Buffer vector to store the projected gradient
- `xₗ`: Lower bounds on `x`
- `xᵤ`: Upper bounds on `x`

# Return

- `πₓ = ||P[x-g] - x||` where `P` denotes the projection onto the box
`[xₗ, xᵤ]` and `||.||` is the `∞`-norm.
"""
function criticality_measure(
    x::AbstractVector{T},
    g::AbstractVector{T},
    gproj::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T}) where T

    project!(gproj, x .- g, xlow, xupp) # gproj ← P[x-g]
    gproj .-= x                         # gproj ← gproj - x
    norm(gproj, Inf)
end

"""
    criticality_measure(x, g, gproj, xₗ, xᵤ)

Computes the criticality measure for problems where linear constraints are polyhedral, which
is the norm of the reduced gradient.

# Arguments

- `x`: Current iterate
- `g`: Gradient of the Augmented Lagrangian at current primal-dual iterate
- `gproj`: Buffer vector to store the projected gradient
- `proj_op`: `Projector` operator to form the projected gradient

# On return

- `πₓ = ||P[x-g] - x||` where `P` denotes the projection onto tangent space of linear
feasible directions at `x` and `||.||` is the `∞`-norm.
"""
function criticality_measure(
    x::AbstractVector{T},
    g::AbstractVector{T},
    gproj::AbstractVector{T},
    proj_op::SubspaceProjector{T}) where T

    mul!(projg, proj_op, g)
    norm(projg, Inf)
end
