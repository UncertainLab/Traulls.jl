using Dates
debug_file = string(now(UTC))*"_debuglv502.out"
debug_io = open(debug_file, "w")
debug = true

"""
    solve(model; kwargs...)

Solve a bound-constrained nonlinear least-squares problem with equality
constraints of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0`

`ℓ ≤ x ≤ u,`

with an iterative Augmented Lagrangian method.

Starting from an initial guess `x₀` and an initial estimate of the vector of
Lagrange multipliers associated to the equality constraints `y₀`,
each new iterate `xₖ₊₁` is an approximate solution, with respect to a tolerance
`ωₖ > 0`, of the subproblem

`minₓ Lₐ(x,yₖ,μₖ) = 1/2 * r(xₖ)ᵀr(xₖ) + c(xₖ)ᵀ[yₖ + μₖ/2 * c(xₖ)]`

`s.t. ℓ ≤ x ≤ u,`

for some penalty parameter `μₖ > 0`, a current estimate of the Lagrange
    multipliers `yₖ` and using `xₖ` as a starting point.

If the new iterate satisfies `||c(xₖ₊₁)||₂ ≤ ηₖ`, for some `ηₖ > 0`, then the
Lagrange multipliers are updated by `yₖ₊₁ = yₖ + μₖc(xₖ)` and the tolerances `ωₖ` and `ηₖ` are tightened.

On the contrary, if xₖ₊₁ fails to satisfies the feasibility inequality, the
iterate is unchanged, i.e. `(xₖ₊₁,yₖ₊₁) = (xₖ,yₖ)` and the minimization of the
subproblem is restarted with
a higher penalty parameter `μₖ₊₁ = τμₖ`, with `τ > 1`. The tolerances `ωₖ` and
`ηₖ` are still reduced but in a weaker maner.

Subproblems are solved by the gradient projection method
see [`projected_gradient`](@ref)).

This solver works in double relative precision.

# Arguments

- `model::BoxCnls`: Encodes the model of the problem to be solved (see [`BoxCnls`](@ref)).

# Keyword Arguments

## Augmented Lagrangian parameters

- `mu0::Float64`: Initial penalty parameter (default: `10.0`)
- `tau::Float64`: Increase factor for the penalty parameter (default: `100.0`)
- `omega0::Float64`: Constant to set the initial criticality tolerance
(default: `1.0`)
- `eta0::Float64`: Constant to set the initial feasibility tolerance
(default: `1.0`)
- `feas_atol::Float64`: Absolute olerance for feasibility of equality
constraints (default: `1e-6`)
- `crit_tol::Float64`: Relative tolerance for criticality (default: `1e-7`)
- `k_crit::Float64`: Positive constant used to initialize and update the
subproblem criticality tolerance in the case of poor improvement of the
feasibility (default: `1.0`)
- `k_feas::Float64`: Positive constant used to initialize and update the
subproblem feasibility tolerance in the case of poor improvement of the
feasibility (default: `0.1`)
- `beta_crit::Float64`: Positive constant used to reduce the subproblem
criticality tolerance in the case of good improvement of the feasibility
(default: `1.0`)
- `beta_feas::Float64`: Positive constant used to reduce the subproblem
feasibility tolerance in the case of good improvement of the feasibility (default: `0.9`)

## Trust region parameters

- `accept_treshold::Float64`: Threshold for accepting a step (default: `0.25`)
- `increase_treshold::Float64`: Threshold for very successful steps in order
to extend the trust region (default: `0.75`)
- `decrease_factor::Float64`: Reducing factor of the trust region
(default: `0.5`)
- `increase_factor::Float64`: Extension factor of the trust region
(default: `2.5`)

## Solver related constants

- `kappa_step::Float64`: Constant to define the tolerance for the projection
gradient method  (default: `0.1`)
- `kappa_cg::Float64`: Constant to define the tolerance for the projected
conjugate gradient method (default: `0.1`)
- `mu_max::Float64`: maximum value of the penalty parameter (default: `1e6`)
- `max_outer_iter`: Maximum number of outer iterations, i.e. number of
minimization of the Augmented Lagrangian (default: `200`)
- `max_inner_iter`: Maximum number of iterations when solving each subproblem
with the gradient projection method (default: `100`)
- `max_cg_iter`: Maximum number of conjugate gradient iterations (default: `50`)

## Miscellaneous

- `output_file_name`: Name of the output file for logging (default: `""` which
makes `stdout` the default output stream)
- `verbose`: Boolean. If set to `true`, execution and iterations detail are
printed into the output file (default: false)

# On return

Returns the solution vector and additional information encoded in a
[`PrimalDualSolution`](@ref).
"""
function solve(
    model::CnlsModel;
    mu::Float64 = 10.0,
    tau::Float64 = 10.0,
    omega0::Float64 = 1.0,
    eta0::Float64 = 1.0,
    min_tol_feas::Float64 = 1e-7,
    min_reltol_crit::Float64 = 1e-7,
    k_crit::Float64 = 1.0,
    k_feas::Float64 = 0.1,
    beta_crit::Float64 = 1.0,
    beta_feas::Float64 = 0.9,
    accept_treshold::Float64 = 0.25,
    increase_treshold::Float64 = 0.75,
    decrease_factor::Float64 = 0.5,
    increase_factor::Float64 = 2.5,
    neg_ratio_factor::Float64 = 0.0625,
    kappa_step::Float64 = 0.1,
    kappa_cg::Float64 = 0.1,
    hessian_approx::HessianApprox = gn,
    mu_max::Float64 = 1e6,
    max_iter::Int = 200,
    max_inner_iter::Int = 200,
    max_cg_iter::Int = 50,
    output_file_name::String="",
    verbose::Bool=false,
    inner_verbose::Bool=false)

    global debug_io
    global debug
    # Sanity checks on arguments
    # Trust region parameters
    !(0 < accept_treshold <= increase_treshold < 1 &&
    0 < decrease_factor < 1 < increase_factor) &&
    error("ArgumentError: trust regions parameters are not valid")

    # Prepare output stream to log iteration detail
    output_io = (output_file_name == "" ? stdout : open(output_file_name,"w"))

    # Dimensions of the problem
    n, nres, ncons = model.n, model.nres, model.ncons
    lincons_present = model.nlincons > 0

    # Make initial point feasible and form subspace projector operator
    x = model.x
    proj_op = initial_point_and_projector!(model, x, Val(lincons_present))
    xlow, xupp = model.xlow, model.xupp

     # Allocate memory for buffer vectors involved in inner minimization
    inner_workspace = Workspace(Float64, n, nres ,ncons)

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
    hess_op = @match hessian_approx begin
        $gn     => GN(J,C,mu)
        $sr1    => SR1(J,C,mu)
    end

    # Set up trust region
    tr = TrustRegion(accept_treshold, increase_treshold, decrease_factor,
    increase_factor, neg_ratio_factor)

    # Set up tolerances
    reltol_crit, tol_feas = initial_tolerances(mu, omega0, eta0, k_crit, k_feas)

    # debug && @printf(debug_io, "[solve] reltol_crit = %.4e ; tol_feas = %.4e\n", reltol_crit, tol_feas)
    # Initial values of objective, feasibility and criticality
    g = J'*rx + C'*y
    fx = dot(rx,rx)
    model.counters.nobj_eval += 1
    feas_measure = norm(cx, Inf)
    gproj = inner_workspace.proj_g

    pix = lincons_present ?
        criticality_measure(x, g, gproj, proj_op) :
        criticality_measure(x, g, gproj, xlow, xupp)

    # tol_scale_factor = max(1, norm(g, Inf))
    tol_crit = min_reltol_crit * (1 + pix)

    solved = feas_measure <= min_tol_feas && pix <= tol_crit

    iter = 1

    # verbose && print_boconls_header(n,nres,ncons,xlow,xupp,min_reltol_crit,min_tol_feas,
    #                                 tau;io=output_io)
    # verbose && print_tr_header(tr;io=output_io)

    verbose && print_traulls_header(model, fx, feas_measure, pix, min_reltol_crit, min_tol_feas, tau, tr; io=output_io)

    # Set counters
    reset!(model.counters)
    start_time = time()

    while !solved && iter <= max_iter

        debug && println(debug_io, "\n\n[solve] outer iter $iter", @sprintf(" relative tolerance : %.6e", reltol_crit))

        # debug && println(debug_io, "[solve] current solution : ", x)
        # debug && println(debug_io, "[solve] current multipliers : ", y, " ; current penalty parameter : ", mu)
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
            kappa_step,
            kappa_cg,
            hessian_approx,
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

            debug && @printf(debug_io, "\n[solve] ||λₖ₊₁ - λₖ|| = %.3e ", norm(cx*mu, Inf))
            # Update Lagrange multipliers
            first_order_multipliers!(y, cx, mu)

            # Evaluate termination status
            g .= J'*rx + C'*y # Lagrangian gradient

            norm_proj_gradlag = lincons_present ?
                criticality_measure(x, g, gproj, proj_op) :
                criticality_measure(x, g, gproj, xlow, xupp)

            debug && @printf(debug_io, "\n[solve] after inner exit πx = %.3e", pix)
            # tol_scale_factor = max(1, norm(g, Inf))
            tol_scale_factor = 1 + norm(g, Inf)

            solved = feas_measure <= min_tol_feas &&
                norm_proj_gradlag <= tol_crit



            if !solved
                # Update the iterate, multipliers and decrease tolerances
                # Penalty parameter is unchanged
                reltol_crit = max(reltol_crit / mu^beta_crit, min_reltol_crit)
                tol_feas = max(tol_feas / mu^beta_feas, min_tol_feas)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances
            # Iterate and multipliers are unchanged
            mu = min(mu_max, tau * mu)
            reltol_crit = max(omega0 / mu^k_crit, min_reltol_crit)
            tol_feas = max(eta0 / mu^k_feas, min_tol_feas)
        end

        verbose && print_outer_iteration(iter, fx, feas_measure, mu, pix, Val(update_multipliers); io=output_io)

        iter += 1

    end

    solving_status = if solved
        first_order_critical
        elseif feas_measure <= min_tol_feas
        feasible_non_critical
        else
        infeasible_non_critical
    end

    elapsed_time = time() - start_time # Save execution time
    model.counters.niter_outer = iter  # Save number of outer iterations

    results = TraullsResults(x, y, fx, feas_measure, pix, solving_status, model.counters,
                             elapsed_time)

    verbose && print(output_io, results)
    # Close output stream
    output_file_name != "" && close(output_io)
    # close(debug_io)
    return results

end

"""
    solve_subproblem!(model, args...)

Solves the outer iteration subproblem

`minₓ Lₐ(x,y,μ) = 1/2 * r(x)ᵀr(x) + c(x)ᵀ[y + μ/2 * c(x)]`

`s.t. ℓ ≤ x ≤ u,`

using the gradient projection method with trust region.

The starting point `x₀` and optimality tolerance `ω` are given. The Lagrange
multipliers `y` and penalty parameter `μ` are fixed.

At iteration `k`, a quadratic model of the objective function around `xₖ` is
formed by

`qₖ(s) = 1/2 sᵀHₖs + sᵀgₖ,`

with `gₖ = ∇ₓLₐ(xₖ,y,μ)` and `Hₖ ≈ ∇²ₓₓ Lₐ(xₖ,y,μ)`.

The step computation consists into approximately solving the quadratic program

`minₛ qₖ(s)`

`s.t. ℓ ≤ xₖ + s ≤ u`

` ||s|| ≤ Δₖ,`

where `Δₖ` is the trust region radius and `||.||` denotes the `∞`-norm
`||x|| = maxᵢ |xᵢ|`. Because `||x|| ≤ Δₖ ⟺ -Δₖ ≤ xᵢ ≤ Δₖ` for all `i`,
the feasible domain for the step can actually be formulated as the box

`Bₖ = [max(-Δₖe, ℓ-x), min(Δₖe, u-x)]`, with `e = (1,...,1)`.

# Solving the QP

## Cauchy point

We start by finding the first local minimizer of the model along the projected
gradient path

`s(t) = Pₖ[xₖ - tgₖ] - xₖ` for  `t ≥ 0,`

`Pₖ` denoting the projection over the feasible domain `Bₖ`.
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

Here, `||.||` denotes the euclidean norm.

## Stopping criteria

The minimization process is stopped once there is an iterate `xₖ` such that

`|| P[xₖ - gₖ] - xₖ || ≤ ω`,

where `P` here denotes the projection operator onto the initial feasible box
`[ℓ,u]`.
This quantity measures how close a point is from first-order criticality.

# Arguments

- `model::BoxCnls{T}`: Structure encoding the original constrained nonlinear
least-squares problem to be solved
- `x::AbstractVector{T}`: Starting point the the outer iteration
- `xlow::AbstractVector{T}`: Lower bounds on the variables
- `xupp::AbstractVector{T}`: Upper bounds on the variables
- `y::AbstractVector{T}`: Current estimation of the Lagrange multipliers
- `mu::T`: Penalty parameter
- `rx::AbstractVector{T}`: Residuals evaluated at `x`
- `cx::AbstractVector{T}`: Equality constraints evaluated at `x`
- `J::AbstractMatrix{T}`: Jacobian of the residuals evaluated at `x`
- `C::AbstractMatrix{T}`: Jacobian of the equality constraints evaluated at `x`
- `g::AbstractVector{T}`: Gradient of the Augmented Lagrangian at `x`
- `tr::TrustRegion`: Encodes the trust region constraint and associated
- `omega_crit::T`: Optimality tolerance
constants
- `kappa_step::T`: Constant used to define the stopping criteria of the
 gradient projection method
- `kappa_cg::T`: Constant used to define the stopping criteria of the
conjugate gradient iterations
- `max_iter::Int`: maximum number of iterations to solve the outer iteration
subproblem
- `max_cg_iter::Int`: maximum number of uses of the conjugate gradient method
- `verbose::Bool=false`: Boolean to log details into a input/output stream
- `io::IO=stdout`: input/output stream (default is `stdout`)
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
    kappa_step::T,
    kappa_cg::T,
    hessian_approx::HessianApprox,
    max_iter::Int,
    max_cg_iter::Int,
    workspace::Workspace{T};
    verbose::Bool=false,
    io::IO=stdout) where T

    global debug_io
    global debug
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
    # debug && println(debug_io, "[solve_subproblem] ∇Lₐ avant calcul: ", g)
    alx = al_objgrad!(rx, cx, y, mu, J, C, g)
    # debug && println(debug_io, "[solve_subproblem] ∇Lₐ apres calcul: ", g)
    model.counters.nalgrad_eval += 1
    model.counters.nalobj_eval += 1
    # Reset Hessian approximation and projector operator
    @match hessian_approx begin
        $gn     => reset_hessian!(hess_op,J,C,mu)
        $sr1    => reset_hessian!(hess_op,J,C,mu)
    end

    reset_projector!(proj_op)

    # Initialize trust region
    set_initial_radius!(tr,g)

    # Prepare for inner minimization loop
    # TODO: add computation of criticality measure for polyhedral problem
    # (when lincons_present = true)
    pix = lincons_present ?
        criticality_measure(x, g, gproj, proj_op) :
        criticality_measure(x, g, gproj, xlow, xupp)

    # tol_scale_factor = max(1, norm(g, Inf))
    tol_crit = reltol_crit * (1 + norm(g, Inf))
    debug && @printf(debug_io, "\n[solve_subproblem] effective tolerance = %.4e", tol_crit)
    solved = pix <= tol_crit
    # println(debug_io, "[solve_subproblem] Initial solved status: $solved")
    short_circuit = false

    iter = 1

    verbose && print_inner_iter_header(io)

    while !solved && iter <= max_iter && !short_circuit

        debug && println(debug_io,"\n[solve_subproblem!] *** inner iter $iter *** ")
        debug && @printf(debug_io, "\n[solve_subproblem!] ||∇f|| = %.6e\n", norm(g))

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
            kappa_step,
            kappa_cg,
            workspace)



        # Check of the trial point is undistinguishable from current solution or
        # if the radius is too small

        # short_circuit = check_stalling(s,x,radius)
        short_circuit = false

        # debug && println(debug_io, "[solve_subproblem] max(|sᵢ|/|xᵢ|) = ", maximum(abs.(s ./ (x .+ 1))))

        if short_circuit continue end

        # Evaluate the objective at trial point
        x .+= s
        residuals!(model, rx, x)
        nlconstraints!(model, cx, x)
        alx = al_obj(rx,cx,y,mu)
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

        # debug && println(debug_io, "[solve_subproblem] |fₖ₊₁ - fₖ|/ |fₖ| = ", abs(alx - alx_prev) / abs(alx_prev))

        # Compute the ratio actual reduction / predicted reduction
        ratio = step_ratio(alx_prev, alx, pred)

        if accept_step(tr, ratio)

            # Evaluate first derivative at trial point

            jac_residuals!(model, J, x)
            jac_nlconstraints!(model, C, x)
            al_grad!(rx, cx, y, mu, J, C, g)
            model.counters.nalgrad_eval += 1

            # Update Hessian approximation
            if hessian_approx == gn
                # Gauss-Newton case
                update_hessian!(hess_op, J, C)

            else
                # Quasi Newton update
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

        debug && @printf(debug_io, "\n[solve_subproblem] criticality after step computation : %.3e", pix)

        # tol_scale_factor = max(1, norm(g, Inf))
        # tol_scale_factor = 1 + norm(g, Inf)
        solved = pix <= tol_crit
        iter += 1
    end

    # debug && println(debug_io, "[solve_subproblem] Number of inner iterations: $iter")
    # Indicate inner iteration finish
    verbose && println(io,'='^92)
    # Save number of inner iterations
    model.counters.niter_inner += iter

    return pix
end

"""
    projected_gradient!(x,g,H,xₗ,xᵤ,Δ,max_cg_iter,κₛ,κᵪ)

Approximately solves the quadratic program

`minₛ 1/2 sᵀHs + sᵀg`

`s.t. xₗ ≤ x + s ≤ xᵤ`

`||s|| ≤ Δ`

by the gradient projection method.

In the QP model, `||.||` denotes the `∞`-norm `||s|| = maxᵢ |sᵢ|`.

# Arguments

- `x::AbstractVector{T}`: Current iterate
- `g::AbstractVector{T}`: Gradient of the Augmented Lagrangian at `x`
- `H::ALHessian{T}`: Approximation of the Hessian of the Augmented Lagrangian at
`x`
- `xₗ::AbstractVector{T}`: Lower bounds on `x`
- `xᵤ::AbstractVector{T}`: Upper bounds on `x`
- `Δ::T`: Trust region radius
- `max_cg_iter::Int`: Number of maximum uses of the conjugate gradient method
- `κₛ::T`: Positive constant used to define the convergence criteria
relative of the gradient projection method
- `κᵪ::T`: Positve constant used to define the convergence criteria of
the conjugate gradient method

# On return

- `s::AbstractVector{T}`: This argument is modified in place and contains the trial step
- `pred::T`: Reduction of the quadratic model after taking step `s`

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
    kappa_step::T,
    kappa_cg::T,
    workspace::Workspace{T}) where T


    # Buffers
    Hs = workspace.hess_vec
    s_low, s_upp = workspace.step_low, workspace.step_upp
    w = workspace.search_dir
    r, v, p = workspace.r, workspace.v, workspace.p

    # Bounds the step  on the search direction
    slow .= (t -> max(-radius, t)).(xlow-x) .- s
    supp .= (t -> min(radius, t)).(xupp-x) .- s
    # Reset active constraints
    reset_projector!(proj_op)

    cauchy_step!(x,
                 s,
                 g,
                 xlow,
                 xupp,
                 slow,
                 supp,
                 hess_op,
                 proj_op,
                 gproj)

    # Update the set of fixed variables (implicitly updates the null space matrix Z)

    # Set up for conjugate gradient iterations
    mul!(Hs, hess_op, s)
    b = workspace.cg_rhs
    b .= Hs .+ g

    quasi_optimal = false
    cg_stop = false
    iter = 1

    while !quasi_optimal && !cg_stop && iter <= max_cg_iter && !saturated_subspace(proj_op)

        cg_status = pcg!(
            b,
            hess_op,
            proj_op,
            s,
            slow,
            supp,
            r,
            v,
            p,
            Hs,
            kappa_cg)


        # Prepare for next CG iterations
        mul!(Hs, hess_op, s) # form Hs
        b .= Hs .+ g

        # Compute norms of reduced gradients ||Zᵀg|| and ||Zᵀ(Hs+g)||

        # TODO: in-place computations for norms of reduced gradients
        norm_reduced_g = norm(proj_op * g)
        norm_reduced_gnext = norm(proj_op * b)

        # Stop if the step provides sufficient decrease in the reduced gradient
        quasi_optimal = norm_reduced_gnext <= kappa_step * norm_reduced_g

        # Stop if negative curvature encountered or if the step lies at the trust region
        # boundary
        cg_stop = cg_status == negative_curvature || cg_status == on_trust_region

        # Identify the newly active bounds
        update_active_set!(s, x, xlow, xupp, radius, proj_op)

        iter += 1
    end

    # Predicted reduction of the model taking step s
    pred = dot(g,s) + 0.5*dot(s,Hs)

    return pred
end

# """ cauchy_step!(x,g,H,ℓ,u,Δ)

# Compute a Cauchy step that provides a sufficient reduction of the quadratic
# model `q(s) = <s,Hs> + <g,s>`.

# The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the
# projected gradient step `P(x-t*g) - x` with `P` denoting the projection over
#  `{v |  max(-Δe,ℓ-x) ≤ v ≤ min(Δe,u-x)}`.

# This method finds the first local minimum of the quadratic model along the
# projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))`
# on `[0, ∞)`.

# The associated Cauchy step is computed in place into vector `s`
# Returns the `BitAbstractVector{T}` `fix_vars` that encodes the indices of active bounds
# at the Cauchy point `x + s`.

# Follows the procedure of algorithm 17.3.1 from Trust Regions Methods
# (Conn, Gould and Toint, SIAM, 2000).
# """
# function cauchy_step!(
#     x::AbstractVector{T},
#     s::AbstractVector{T},
#     g::AbstractVector{T},
#     d::AbstractVector{T},
#     hess_op::ALHessian{T},
#     proj_op::CoordinateSubspaceProjector{T},
#     Hd::AbstractVector{T},
#     xlow::AbstractVector{T},
#     xupp::AbstractVector{T},
#     radius::T) where T

#     global debug
#     global debug_io

#     epsmch = eps(T)
#     epsrel = sqrt(epsrel)
#     eps_slope = 1e-10
#     eps_curv = 1e-10

#     n = size(x,1)
#     # Accumulated Cauchy step
#     s .= T(0)


#     # Breakpoints values and group indices
#     breakpoints, grp_idx = sort_breakpoints(x, g, xlow, xupp, radius)
#     prev_tb = 0.0
#     d .= -g


#     # Handle the case where the first breakpoint is zero
#     # Happens when bounds are active at x
#     if iszero(breakpoints[1])

#         popfirst!(breakpoints)                      # get rid of breakpoint tb = zero
#         first_active_index = popfirst!(grp_idx)
#         # debug && println(debug_io, "[cauchy_step] components with breakpoint at 0 = $(size(first_active_index,1))")
#         update_projector!(proj_op,first_active_index)
#         mul!(d,proj_op,-g)
#     end

#     # Find fix components

#     for i = 1:n
#         # Variable at lower bound
#         if x[i] <= xlow[i] + abs(xlow[i])*epsrel && d[i] <= epsrel
#             set_active!(proj_op, i)
#         # Variable at upper bound
#         elseif x[i] >= xupp[i] - abs(xupp[i])*epsrel && d[i] >= -epsrel
#             set_active!(proj_op, i)
#         end
#     end
#     # Variables with ≈ 0 direction
#     zero_dir = findall(t -> abs(t) <= epsrel, d)
#     set_active!(proj_op, zero_dir)

#     # Update direction
#     mul!(d, proj_op, -g)

#     # TODO: modify breakpoint finding method

#     gtd = dot(g,d)
#     mul!(Hd,hess_op,d)

#     for (i, tb) in enumerate(breakpoints)

#         # Compute slope and curvature
#         phi_p = gtd + dot(s,Hd)
#         phi_pp = dot(d,Hd)

#         # Study the current interval [prev_tb, tb)
#         delta_t = (phi_pp > 0 ? -phi_p / phi_pp : 0.0)
#         l_interval = tb - prev_tb

#         if phi_p >= 0
#             break
#         elseif phi_pp > 0 && delta_t < l_interval # local minimum at t = tb - phi_p / phi_pp
#             s .+= delta_t .* d
#             break
#         end

#         # No local minimum in [prev_tb, tb)
#         # Prepare for the next interval
#         prev_tb = tb
#         newly_active = grp_idx[i]
#         update_projector!(proj_op, newly_active)

#         s .+= d .* l_interval
#         mul!(d, proj_op, -g)
#         gtd = dot(g, d)
#         mul!(Hd, hess_op, d)
#     end

#     # Set free variables fixed because of zero direction
#     set_free!(proj_op, )
#     return
# end


# """
#     cauchy_step!(x,g,H,proj_op,ℓ,u,Δ)

# Compute a Cauchy step that provides a sufficient reduction of the quadratic model
# `q(s) = <s,Hs> + <g,s>`.

# The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the
# projected gradient step `P(x-t*g) - x` with `P` denoting the projection over
# `{v | Av = 0 and max(-Δ,ℓ) ≤ x + v ≤ min(Δ,u)}`.

# This method finds the first local minimum of the quadratic model along the
# projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

# The associated Cauchy step is computed in place into vector `s`
# Returns the `BitAbstractVector{T}` `fix_vars` that encodes the indices of active bounds
# at the Cauchy point `x + s`.

# Follows the procedure of algorithm 17.3.1 from Trust Regions Methods
# (Conn, Gould and Toint, SIAM, 2000).

# # Arguments

# - `x::Vector`: current iterate
# - `s::Vector`: buffer vector for the Cauchy step
# - `g::Vector`: gradient of the augmented Lagrangian at current point
# - `d::Vector`: buffer vector the the projected steepest direction
# - `hess_op`: Hessian approximation of type `ALHessian` at current point
# - `proj_op`: [`SubspaceProjector`](@ref) operator to project the negative gradient on
# tangent spaces
# - `Hd::Vector`: buffer vector for the
# - `xlow::Vector`: lower bounds on the variables `x`
# - `xupp::Vector`: upper bounds on the variables `x`
# - `radius::T`: Current trust region radius

# # On return

# - `s` argument modified in place with components set to the Cauchy step

# """
# function cauchy_step!(
#     x::AbstractVector{T},
#     s::AbstractVector{T},
#     g::AbstractVector{T},
#     d::AbstractVector{T},
#     hess_op::ALHessian{T},
#     proj_op::SubspaceProjector{T},
#     Hd::AbstractVector{T},
#     xlow::AbstractVector{T},
#     xupp::AbstractVector{T},
#     radius::T) where T

#     (m,n) = size(proj_op.workspace_mat.eqmat)

#     # Initial projected steepest direction
#     mul!(d, proj_op, -g)

#     # Check if they are bounds active at x
#     prev_tb = 0
#     initial_fixed = initial_active_bounds(x, d, xlow, xupp)

#     if !isempty(initial_fixed)
#         update_projector!(proj_op, initial_fixed)
#     end

#     # Update the projection
#     mul!(d, proj_op, -g)

#     # Prepare the first interval
#     # Find the first breakpoint
#     tb, idx = next_breakpoint(d, s, xlow, xupp, radius, proj_op.workspace_mat.fixvars)

#     # Form gᵀd and Hd
#     gtd = dot(g,d)
#     mul!(Hd, hess_op, d) # Hd ← H*d

#     # Search while constraints can be added to the active set or breakpoints exist
#     while !saturated_subspace(proj_op) && !isempty(idx)

#         # Compute slope and curvature
#         phi_p = gtd + dot(s,Hd)
#         phi_pp = dot(d,Hd)

#         # Study the current interval [prev_tb, tb)
#         delta_t = (phi_pp > 0 ? -phi_p / phi_pp : 0.0)
#         l_interval = tb - prev_tb

#         if phi_p >= 0
#             # local minimum at previous breakpoint
#             break
#         elseif phi_pp > 0 && delta_t < l_interval
#             # local minimum at t = tb - phi_p / phi_pp
#             s .+= delta_t .* d
#             break
#         end

#         # No local minimum in [prev_tb, tb)
#         # Update accumulated step
#         s .+= d .* l_interval

#         # Compute the projected direction on the next interval
#         update_projector!(proj_op,idx)
#         mul!(d, proj_op, -g)

#         # Prepare for the next interval
#         gtd = dot(g,d)
#         mul!(Hd, hess_op, d)

#         prev_tb = tb
#         # Find next breakpoint
#         tb, idx = next_breakpoint(d,s,d_low,d_upp,proj_op.workpsace_mat.fixvars)

#     end

#     return
# end

# """ next_breakpoint(d,s,dₗ,dᵤ,fix_bounds)

# Finds the smallest scalar `θ` such that one or more components not in `fix_bounds`
# of `s + θ*d` lie at one of their bounds `dₗ` or `dᵤ`.

# Returns the scalar `θ` and `idx`, the index of the components that becomes active.
# """
# function next_breakpoint(
#     d::AbstractVector{T},
#     s::AbstractVector{T},
#     xlow::AbstractVector{T},
#     xupp::AbstractVector{T},
#     radius::T,
#     fix_bounds::BitVector;
#     atol::T=sqrt(eps(T))) where T

#     theta = Inf # current breakpoint value
#     idx = []    # list of bounds indicices becoming active at theta

#     for i in axes(d,1)
#         if !fix_bounds[i]

#             theta_try = if d[i] < -atol
#                 (max(xlow[i] - x[i], -radius) - s[i]) / d[i]
#             elseif d[i] > atol
#                 (min(xupp[i] - x[i], radius) - s[i]) / d[i]
#             else
#                 Inf
#             end

#             also_bp = abs(theta_try-theta) < atol

#             if also_bp
#                 push!(idx,i)

#             elseif !also_bp && theta_try < theta
#                 theta = theta_try
#                 idx = [i]
#             end
#         end
#     end
#     return theta, idx
# end

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
# Forms and returns the operator computing projections on reduced subspaces
function initial_point_and_projector!(
    model::CnlsModel{T},
    x::AbstractVector{T},
    ::Val{true};
    tol::T=sqrt(eps(T))) where T

    A, b = model.linmat, model.linrhs
    n, m = model.n, model.nlincons
    xlow, xupp = model.xlow, model.xupp

    solve_linfeas_pb!(A, x0, b, xlow, xupp, m, n)

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

    @variable(feas_model, lb[i] <= x[i=1:n] <= lb[i], start = x0[i])
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

Computes the criticality measure used to measure if a primal-dual solution
`(x,y)` is a first-order critical point or not.

# Arguments

- `x::AbstractVector{T}`: Current iterate
- `g::AbstractVector{T}`: Gradient of the Augmented Lagrangian at current primal-dual
iterate `(x,y)`
- `xₗ::AbstractVector{T}`: Lower bounds on `x`
- `xᵤ::AbstractVector{T}`: Upper bounds on `x`
- `p::T`: Nature of the norm computed (default is `Inf`).

# Return

- `πₓ = ||P[x-g] - x||` where `P` denotes the projection onto the box
`[xₗ, xᵤ]` and `||.||` is the `p`-norm for some `p > 1`.

In practice, either the `ℓ₂` or `∞` norms are used.
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

# The measure computed is the norm of the projection of the steepest direction
# on the tangent space at a given point. That information is encoded inside the
# `proj_op` argument.
function criticality_measure(
    x::AbstractVector{T},
    g::AbstractVector{T},
    projg::AbstractVector{T},
    proj_op::SubspaceProjector{T}) where T

    mul!(projg, proj_op, g)

end
