"""
    PolyhedralCnls <: AbstractCnlsModel

Mutable struct encoding a constrained nonlinear least-squares problem of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

`Ax = b`

`ℓ ≤ x ≤ u.`

Functions `r`, `h` and `g` are two times continuously differentiable, nonlinear 
and potentially non convex.

Inequality constraints are implicitly converted into equalities by adding slack
variables, so that `g(x) ≥ 0` becomes `g(x) - z = 0, z ≥ 0.`

'A' is a full line rank matrix of size `m × n`, with `m < n`.

Bounds on the variables `ℓ` and `u` can be set to `±∞`.

**Attributes** 

* `res`: function evaluating the residuals

* `nleq`: function evaluating the nonlinear equality constraints 

* `nlineq`: function evaluating the nonlinear inequality constraints 

* `jac_res`: function evaluating the jacobian matrix of the residuals

* `jac_nleq`: function evaluating the jacobian matrix of the nonlinear equality
 constraints

* `jac_nlineq`: function evaluating the jacobian matrix of the nonlinear 
inequality constraints 

* `eqmat`: matrix of the linear equality constraints 

* `eqrhs`: right handside vector of the linear equality constraints  

* `n::Int`: number of variables (size of `x`)

* `x_low::Vector`: vector of lower bounds on the parameters 

* `x_upp::Vector`: vector of upper bounds on the parameters 

* `x::Vector`: initial guess for the solution

* `n_slack::Int`: number of slack variables (size of `z`)

* `m::Int`: number of residuals (size of `r(x)`)

* `p::Int`: number of nonlinear equality constraints (size of `c(x)`)


**Note**

When instantiating a `PolyhedralCnls`, arguments for evaluation functions 
`res`, `jac_res` etc. must be functions of a single `Vector` argument of size
`n` and return a `Vector` or `Matrix` of appropriate dimensions.

For instance, evaluating the residuals must be done by calling `r(x)` and the
output must be a `Vector` of size `m`. Similarly, `c(x)` must be of size `p`, 
`J(x)` of size `m × n` and `C(x)` of size `p × n`.
"""
mutable struct PolyhedralCnls <: AbstractCnlsModel
    res
    nleq
    nlineq
    jac_res
    jac_nleq
    jac_nlineq
    eqmat::Matrix
    eqrhs::Vector
    x_low::Vector
    x_upp::Vector
    x::Vector
    n::Int
    n_slack::Int
    m::Int
    p::Int
end

"""
    PolyhedralCnls(r,h,g,jac_r,jac_h,jac_g,A,b,low,upp,n_var,m,p_eq,p_ineq)

Constructor for the [`BoxCnls`](@ref) structure.

Encodes a nonlinear least-squares problems of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

`Ax = b`

`low ≤ x ≤ upp.`

Nonlinear inequality constraints are converted as equality constraints by
adding slack variables`g(x) - u = 0` with `u ≥ 0`.
 
**Arguments**

* `r`: Function evaluating the residuals
* `h`: Function evaluating the nonlinear equality constraints
* `g`: Function evaluating the nonlinear inequality constraints
* `jac_r`: Function evaluating the Jacobian of the residuals
* `jac_h`: Function evaluating the Jacobian of the nonlinear equality constraints
* `jac_g`: Function evaluating the Jacobian of the nonlinear inequality constraints
* `A`: MAtrix of the linear equality constraints 
* `b`: Right handside vector of the linear equality constraints 
* `low`: Lower bounds on the variables
* `upp`: Upper bounds on the variables
* `x_start`: Initial solution
* `n_var`: Number of variables
* `m`: Number of residuals
* `p_eq`: Number of nonlinear equality constraints
* `p_ineq`: Number of nonlinear inequality constraints
"""
function PolyhedralCnls(
    r,
    h,
    g,
    jac_r,
    jac_h,
    jac_g,
    A,
    b,
    low,
    upp,
    x_start,
    n_var::Int,
    m::Int,
    p_eq::Int,
    p_ineq::Int)

    println("coucou")

    # Dimensions
    n_slack = p_ineq
    n = n_var + n_slack
    p = p_eq + p_ineq

    # Form linear constraints to take into account slack variables
    x_low = vcat(low, zeros(n_slack))
    x_upp = vcat(upp, fill(Inf,n_slack))

    eqmat = hcat(A,zeros(size(A,1),n_slack))

    return PolyhedralCnls(r,h,g,jac_r,jac_h,jac_g,eqmat,b,x_low,x_upp,x_start,
    n,n_slack,m,p)
end

"""
    PolyhedralCnls(r,c,jac_r,jac_c,A,b,low,upp,n_var,m,p,only_equalities)

Constructor for the [`PolyhedralCnls`](@ref) structure.

Encodes a nonlinear least-squares problems of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0 or c(x) ≥ 0`

`Ax = b`

`low ≤ x ≤ upp.`

Nonlinear inequality constraints are converted as equality constraints by
adding slack variables`c(x) - u = 0` with `u ≥ 0`.
 
**Arguments**

* `r`: Function evaluating the residuals
* `c`: Function evaluating the nonlinear constraints
* `jac_r`: Function evaluating the Jacobian of the residuals
* `jac_c`: Function evaluating the Jacobian of the nonlinear constraints
* `A`: Linear equality constraints matrix 
* `b`: Right handside vector of the linear equality constraints 
* `low`: Lower bounds on the variables
* `upp`: Upper bounds on the variables
* `x_start`: Initial values of the variables
* `n_var`: Number of variables
* `m`: Number of residuals
* `p`: Number of nonlinear constraints
* `only_equalities`: Boolean indicating the nature of the nonlinear constraints.
The latter are treated as equalities if set to `true` and as inequalities if 
set to false
"""
function PolyhedralCnls(
    r,
    c,
    jac_r,
    jac_c,
    A,
    b,
    low,
    upp,
    x_start,
    n_var::Int,
    m::Int,
    p::Int,
    only_equalities::Bool)

    
    return if only_equalities
        PolyhedralCnls(r,c,nothing,jac_r,jac_c,nothing,A,b,low,upp,x_start,
        n_var,0,m,p)
    
    else begin

        # Adjust dimensions, starting point and linear constraints to take into 
        # account the slack variables 
        n_slack = p
        n = n_var + n_slack
        eqmat = hcat(A,zeros(size(A,1),n_slack))
        x_low = vcat(low, zeros(n_slack))
        x_upp = vcat(upp, fill(Inf,n_slack))
        x0 = vcat(x_start,c(x_start))

        PolyhedralCnls(r,nothing,c,jac_r,nothing,jac_c,eqmat,b,x_low,x_upp,x0,n,
        n_slack,m,p) 
    end
    end
end
"""
    residuals!(model::PolyhedralCnls, x::Vector, v::Vector)

Compute the residuals for the given model and input vector `x`, storing the result in `v`.
"""
function residuals!(model::PolyhedralCnls, x::Vector, v::Vector) end

"""
    residuals(model::PolyhedralCnls, x::Vector) 

Return the residuals for the given model and input vector `x` as a new vector.
"""
function residuals(model::PolyhedralCnls,x::Vector)  
    rx = Vector{eltype(x)}(undef,model.m)
    residuals!(model, x, rx)
    return rx
end

"""
    nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) 

Compute the nonlinear constraints for the given model and input vector `x`, storing the result in `v`.
"""
function nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) end

"""
    nlconstraints(model::PolyhedralCnls, x::Vector) 

Return the nonlinear constraints for the given model and input vector `x` as a new vector.
"""
function nlconstraints(model::PolyhedralCnls,x::Vector)  
    cx = Vector{eltype(x)}(undef,model.p)
    nlconstraints!(model, x, cx)
    return cx
end

"""
    jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) 

Compute the Jacobian of the residuals for the given model and input vector `x`, storing the result in matrix `J`.
"""
function jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) end

"""
    jac_residuals(model::PolyhedralCnls, x::Vector) 

Return the Jacobian of the residuals for the given model and input vector `x` as a new matrix.
"""
function jac_residuals(model::PolyhedralCnls, x::Vector)  
    Jx = Matrix{eltype(x)}(undef,model.m, model.n)
    jac_residuals!(model, x, Jx)
    return Jx
end



"""
    jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix) 

Compute the Jacobian of the nonlinear constraints for the given model and input vector `x`, 
storing the result in matrix `C`.
"""
function jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix) end

"""
    jac_nlconstraints(model::PolyhedralCnls, x::Vector) 

Return the Jacobian of the nonlinear constraints for the given model and input vector `x` as a new matrix.
"""
function jac_nlconstraints(model::PolyhedralCnls,x::Vector)  
    Cx = Matrix{eltype(x)}(undef,model.p,model.n)
    jac_nlconstraints!(model, x, Cx)
    return Cx
end

"""
    traulls(model; kwargs...)

Solve a linearly constrained nonlinear least-squares problem with equality
constraints of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0`

`Ax = b`

`ℓ ≤ x ≤ u,`

by an iterative Augmented Lagrangian method.

Functions 'r' and 'c' are two times continuously differentiable, nonlinear and potentially non convex.

'A' is a full line rank matrix of size `m × n`, with `m < n`.

Bounds on the variables `ℓ` and `u` can be set to `±∞`.

Starting from an initial guess `x₀` and an initial estimate of the vector of
Lagrange multipliers associated to the equality constraints `y₀`,
each new iterate `xₖ₊₁` is an approximate solution, with respect to a tolerance
 `ωₖ > 0`, of the subproblem

`minₓ Lₐ(x,yₖ,μₖ) = 1/2 * r(xₖ)ᵀr(xₖ) + c(xₖ)ᵀ[yₖ + μₖ/2 * c(xₖ)]`²`

`s.t. Ax = b`

`s.t. ℓ ≤ x ≤ u,`

for some penalty parameter `μₖ > 0`, a current estimate of the Lagrange
multipliers `yₖ` and using `xₖ` as a starting point.

If the new iterate satisfies `||c(xₖ₊₁)||₂ ≤ ηₖ`, for some `ηₖ > 0`, then the
Lagrange multipliers are updated by `yₖ₊₁ = yₖ + μₖc(xₖ)` and the tolerances
`ωₖ` and `ηₖ` are tightened.

On the contrary, if xₖ₊₁ fails to satisfies the feasibility inequality, the
iterate is unchanged, i.e. `(xₖ₊₁,yₖ₊₁) = (xₖ,yₖ)` and the minimization of the
 subproblem is restarted with
a higher penalty parameter `μₖ₊₁ = τμₖ`, with `τ > 1`. The tolerances `ωₖ` and
`ηₖ` are still reduced but in a weaker maner.

Subproblems are solved by the gradient projection method (see [`projected_gradient`](@ref)).

This solver works in double relative precision.

# Arguments

- `model::PolyhedralCnls`: Encodes the model of the problem to be solved


# Keyword Arguments

- `x::Vector`: initial guess for the variables (default: `zeros(n)`)
- `output_file_name`: name of the output file for logging (default: `""` which
makes `stdout` the default output stream)
- `verbose`: Boolean. If set to `true`, execution and iterations detail are
printed into the output file (default: false)

## Augmented Lagrangian parameters

- `mu::Float64`: initial penalty parameter (default: `10.0`)
- `tau::Float64`: increase factor for the penalty parameter (default: `100.0`)
- `omega0::Float64`: constant to set the initial criticality tolerance
(default: `1.0`)
- `eta0::Float64`: constant to set the initial feasibility tolerance
(default: `1.0`)
- `feas_tol::Float64`: tolerance for feasibility of equality constraints
(default: `1e-6`)
- `crit_tol::Float64`: tolerance for criticality (default: `1e-5`)
- `k_crit::Float64`: positive constant used to initialize and update the
subproblem criticality tolerance in the case of poor improvement of the
feasibility (default: `1.0`)
- `k_feas::Float64`: positive constant used to initialize and update the
subproblem feasibility tolerance in the case of poor improvement of the
feasibility (default: `0.1`)
- `beta_crit::Float64`: positive constant used to reduce the subproblem
criticality tolerance in the case of good improvement of the feasibility (default: `1.0`)
- `beta_feas::Float64`: positive constant used to reduce the subproblem
feasibility tolerance in the case of good improvement of the feasibility (default: `0.9`)


## Trust region parameters

- `accept_treshold::Float64`: threshold for accepting a step (default: `0.25`)
- `increase_treshold::Float64`: threshold for very successful steps in order to
extend the trust region (default: `0.75`)
- `decrease_factor::Float64`: reducing factor of the trust region (default: `0.5`)
- `increase_factor::Float64`: extension factor of the trust region (default: `2.5`)


## Other solver related constants

- `kappa_step::Float64`: constant to define the tolerance for the projection
gradient method  (default: `0.1`)
- `kappa_cg::Float64`: constant to define the tolerance for the projected
conjugate gradient method (default: `0.1`)
- `max_outer_iter`: maximum number of outer iterations, i.e. number of
minimization of the Augmented Lagrangian (default: `200`)
- `max_inner_iter`: maximum number of iterations when solving each subproblem
 with the gradient projection method (default: `100`)
- `max_cg_iter`: maximum number of conjugate gradient iterations (default: `50`)

# Return

Returns the solution vector and additional information encoded in a [`PrimalDualSolution`](@ref).
"""
function solve(
    model::PolyhedralCnls;
    x::Vector{Float64}=zeros(model.n),
    mu::Float64 = 10.0,
    tau::Float64 = 10.0,
    omega0::Float64 = 1.0,
    eta0::Float64 = 1.0,
    feas_atol::Float64 = 1e-6,
    crit_rtol::Float64 = 1e-7,
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
    max_iter::Int = 100,
    max_inner_iter::Int = 100,
    max_cg_iter::Int = 50,
    output_file_name::String="",
    verbose::Bool=false)

    
    # Sanity check on parameters
    @assert (0 < accept_treshold <= increase_treshold < 1) && (0 < decrease_factor < 1 < increase_factor) "Invalid trust region paramaters"

    # Local scope evaluation methods 

    function residuals!(model::PolyhedralCnls, x::Vector, v::Vector) 
        v[:] .= model.res(x[1:model.n-model.n_slack])
        return 
    end

    function nlconstraints!(model::PolyhedralCnls, x::Vector, v::Vector) 
        n, n_slack, p = model.n, model.n_slack, model.p
        n_var = n - n_slack
        p_eq = p - n_slack

        x_var = view(x,1:n_var)
        x_slack = view(x,n_var+1:n)

        # Equality constraints components
        if p_eq > 0 v[1:p_eq] .= model.nleq(x_var) end 

        # Inequality constraints transformed into equalities
        if n_slack > 0  v[p_eq+1:end] .= model.nlineq(x_var) .- x_slack end

        return
    end

    function jac_residuals!(model::PolyhedralCnls, x::Vector, J::Matrix) 
        n, n_slack, m = model.n, model.n_slack, model.m
        n_var = n - n_slack

        J[:,1:n_var] .= model.jac_res(x[1:n_var])

        if n_slack > 0
            J[:,n_var+1:end] .= zeros(m,n_slack)
        end
        return
    end

    function jac_nlconstraints!(model::PolyhedralCnls, x::Vector, C::Matrix)
        n, n_slack, p = model.n, model.n_slack, model.p
        n_var = n - n_slack
        p_eq = p-n_slack

        x_var = view(x,1:n_var)

        # Equality constraints components
        if p_eq > 0 
            C[1:p_eq,:] .= hcat(model.jac_nleq(x_var), zeros(p_eq,n_slack)) 
        end

        # Inequality constraints transformed into equalities
        if n_slack > 0 
            C[p_eq+1:end,:] .= hcat(model.jac_nlineq(x_var), Diagonal{Float64}(-I,n_slack)) 
        end

        return
    end


    n, m, p = model.n, model.m, model.p
    x_low, x_upp = model.x_low, model.x_upp

    A = model.eqmat
    chol_aat = cholesky(A*A')

    # Initialize structures
    tr = TrustRegion(accept_treshold, increase_treshold, decrease_factor, increase_factor, neg_ratio_factor)
    proj_op = SubspaceProjector(A,chol_aat)

    # Output stream
    output_stream = output_file_name == "" ? stdout : output_file_name

    verbose && print_boconls_header(n,m,p,x_low,x_upp,omega_rel,feas_tol,tau; io=stream)
    verbose && print_tr_header(tr;io=output_file)

    # Allocation of buffers and first evaluations
    rx = residuals(model, x)
    cx = nlconstraints(model, x)
    J = jac_residuals(model, x)
    C = jac_nlconstraints(model, x)

     # Initial tolerances
    omega_rel, eta = initial_tolerances(mu, omega0, eta0, k_crit, k_feas)
    # Initial Lagrange multipliers
    y = least_squares_multipliers(rx, J, C)

    fx = dot(rx,rx)
    feas_measure = norm(cx,Inf)
    # feas_measure = norm(cx)

    g = al_grad(rx,cx,y,mu,J,C)
    g0 = copy(g) # copy initial gradient for termination criteria

    # TODO: compute a more precise initial criticality measure
    pix = criticality_measure(g0,proj_op)
    first_order_critical = pix <= crit_rtol

    iter = 1


    while !first_order_critical && iter <= max_iter

        verbose && print_outer_iter_header(iter,fx,feas_measure,mu,pix,omega; io=output_stream)

        pix = solve_subproblem(
            model,
            x,
            x_low,
            x_upp,
            proj_op,
            y,
            mu,
            rx,
            cx,
            J,
            C,
            g,
            tr,
            omega_rel,
            kappa_cg,
            hessian_approx,
            max_inner_iter;
            verbose=verbose,
            io=output_stream)

        feas_measure = norm(cx,Inf)

        if feas_measure <= eta
            
            pix0 = criticality_measure(g0,proj_op)
            crit_tol = max(crit_rtol, crit_rtol*pix0)
            first_order_critical = feas_measure <= feas_atol && pix <= crit_tol

            first_order_multipliers!(y,cx,mu)

            if !first_order_critical
                # Update the iterate, multipliers and decrease tolerances (penalty parameter is unchanged)
                omega = max(omega / mu^beta_crit, crit_rtol)
                eta = max(eta / mu^beta_feas, feas_atol)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances (iterate and multipliers are unchanged)
            mu *= tau
            omega = max(omega0 / mu^k_crit, crit_rtol)
            eta = max(eta0 / mu^k_feas, feas_atol)
        end

        iter += 1
        fx  = dot(rx,rx)
    end

    verbose && print_termination_info(iter,x,y,mu,fx,pix,feas_measure;io=stream)
    verbose && close(output_stream)

    PrimalDualSolution(x,y,fx,pix,feas_measure)

    end


"""
    solve_subproblem(model, args...)

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

`s.t. As = 0`

`ℓ ≤ xₖ + s ≤ u`

`||s|| ≤ Δₖ,`

where `Δₖ` is the trust region radius and `||.||` denotes the `∞`-norm
`||x|| = maxᵢ |xᵢ|`. Because `||x|| ≤ Δₖ ⟺ -Δₖ ≤ xᵢ ≤ Δₖ` for all `i`,
the bounds on the step can be simplified as

`s ∈ [max(-Δₖe, ℓ-x), min(Δₖe, u-x)]`, with `e = (1,...,1)`.

# Solving the QP

## Cauchy point

We start by finding the first local minimizer of the model along the projected
gradient path

`s(t) = P[xₖ - tgₖ] - xₖ` for  `t ≥ 0,`

`P` denoting the projection over the feasible domain. The corresponding scalar
defines a Cauchy step that ensures a sufficient reduction of the objective
function. This means that taking the Cauchy step at every iteration is enough to
solve the subproblem.

## Beyond the Cauchy point

In order to provide a better reduction, we then apply the conjugate gradient
method to the subspace where the components corresponding to bounds active at the
 Cauchy point are fixed.

The resulting `sₖ` step is then accepted or rejected depending on the value of
the ratio of the actual reduction over the reduction predicted by the model

`ρ = (Lₐ(xₖ+sₖ,y,μ) - Lₐ(xₖ,y,μ)) / qₖ(sₖ) - qₖ(0)`.

If `ρ ≥ η₁`, where `η₁ ∈ (0,1)` is a given parameter, then the step is accepted
and the radius `Δₖ` is eventually increased. This translates the fact that there
 is a good agreement between the objective function and the model.

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

where `P` here denotes the projection operator onto the subspace that maintains
the active linear constraints active (equalities + active bounds).

This quantity measures how close a point is from first-order criticality.

# Arguments

- `model::BoxCnls`: structure encoding the original constrained nonlinear
least-squares problem to be solved
- `x::Vector`: starting point the the outer iteration
- `x_low::Vector`: lower bounds on the variables
- `x_upp::Vector`: upper bounds on the variables
- `proj_op::SubspaceProjector`: operator involved in computations of projections
onto the linear constraints feasible domain (see [`SubspaceProjector`](@ref))
- `y::Vector`: current estimation of the Lagrange multipliers
- `mu::Float64`: penalty parameter
- `rx::Vector`: residuals evaluated at `x`
- `cx::Vector`: equality constraints evaluated at `x`
- `J::Matrix`: jacobian of the residuals evaluated at `x`
- `C::Matrix`: jacobian of the equality constraints evaluated at `x`
- `g::Vector`: gradient of the Augmented Lagrangian at `x`
- `tr::TrustRegion`: encodes the trust region constraint and associated constants
- `omega_crit::Float64`: optimality tolerance
- `kappa_step::Float64`: constant used to define the stopping criteria of the
gradient projection method
- `kappa_cg::Float64`: constant used to define the stopping criteria of the
conjugate gradient iterations
- `hessian_approx::HessianApprox`: Encodes the type of Hessian approximation used
(see [`HessianApprox`](@ref))
- `max_iter::Int`: maximum number of iterations to solve the outer iteration
 subproblem
- `max_cg_iter::Int`: maximum number of uses of the conjugate gradient method
- `verbose::Bool=false`: boolean to log details into a input/output stream
- `io::IO=stdout`: input/output stream (default is `stdout`)

# On return

- `pix::Float64`: value of the criticality measure at the approximate solution
"""
function solve_subproblem(
    model::PolyhedralCnls,
    x::Vector,
    x_low::Vector,
    x_upp::Vector,
    proj_op::SubspaceProjector,
    y::Vector,
    mu::Float64,
    rx::Vector,
    cx::Vector,
    J::Matrix,
    C::Matrix,
    g::Vector,
    tr::TrustRegion,
    omega_rel::Float64,
    kappa_cg::Float64,
    hessian_approx::HessianApprox,
    max_iter::Int;
    verbose::Bool=false,
    io::IO=stdout)

    # Set dimensions and buffers 
    n, n_slack, p = model.n, model.n_slack, model.p
    x_prev, rx_prev, cx_prev = copy(x), copy(rx), copy(cx)
    
    reset_projector!(proj_op)  # set all bounds as inactive

    # Evaluate objective, first derivatives and Hessian of the AL at current point (x,y)
    alx = al_objgrad!(rx,cx,y,mu,J,C,g)
    g0 = copy(g) # save initial gradient for relative termination criteria

    hess_op = @match hessian_approx begin
        $gn     => GN(J,C,mu)
        $sr1    => SR1(J,C,mu)
    end

    set_initial_radius!(tr,g)

    solved, short_circuit = false, false 

    iter = 1 

    while !solved && iter <= max_iter && !short_circuit

        x_prev .= x 
        rx_prev .= rx 
        cx_prev .= cx
        alx_prev = alx

        radius = tr.radius 

        s, pred = projected_gradient(
            x,
            g,
            hess_op,
            proj_op,
            x_low,
            x_upp,
            radius,
            kappa_cg)

        # Trial point undistinguishable from current solution or too small radius
        
        short_circuit = check_stalling(s,x,radius)

        if short_circuit continue end

        # Evaluate the objective at trial point
        x .+= s
        residuals!(model,x,rx)
        nlconstraints!(model,x,cx)
        alx = al_obj(rx,cx,y,mu)

        # Step taken on the slack variables, if any
        if n_slack > 0
            # Add "magical" step to current point x
            step_slack!(x,y,cx,mu,n_slack,p)

            # Adjust the step vector
            slack_idx = n - n_slack + 1 : n
            ineq_idx = p - n_slack + 1 : p
            s[slack_idx] .= x[slack_idx] .- x_prev[slack_idx] .- s[slack_idx]

            # Update the constraints involving slack variables without evaluating
            cx[ineq_idx] .-= s[slack_idx]

            # Add reduction of the true objective function after taking second
            # step to pred
            pred -= alx
            alx = al_obj(rx,cx,y,mu)
            pred += alx

        end

        # Compute the ratio actual reduction / predicted reduction
        ratio = step_ratio(alx_prev, alx, pred)
        # verbose && println("[solve_subproblem] ared = $ared, pred = $pred, ratio = $ratio")

        if accept_step(tr,ratio)

            # Update gradient and Hessian approximation

            if hessian_approx == gn # Gauss-Newton case
                # Update Jacobians for form next iteration Gauss-Newton
                # approximation
                jac_residuals!(model,x,J)
                jac_nlconstraints!(model,x,C)
                update_hessian!(hess_op,J,C)

                al_grad!(rx,cx,y,mu,J,C,g) # Evaluate gradient

            else # Quasi Newton update
                # Update Jacobians and apply structured SR1 update to
                # second order terms

                # Form right handside of the secant equation

                y_a .= -J'*rx .- C'*(y .+ mu.*cx)
                jac_residuals!(model,x,J)
                jac_nlconstraints!(model,x,C)
                al_grad!(rx,cx,y,mu,J,C,g)
                y_a .+= g

                update_hessian!(hess_op,J,C,y_a,s)

            end

            pix = criticality_measure(g,proj_op)

        else
            x .= x_prev
            rx .= rx_prev
            cx .= cx_prev
            alx = alx_prev
        end

        norm_step = norm(s,Inf)
        # norm_step = norm(s)
        update_radius!(tr,ratio,norm_step)

        verbose && print_inner_iter(iter,alx_prev,norm_step,radius,ratio;io=io)

        solved = begin
            pix0 = criticality_measure(g0,proj_op)
            pix <= max(omega_rel, omega_rel*pix0)
        end

        iter += 1
    end

    return pix
end

"""
    projected_gradient(x,g,H,xₗ,xᵤ,Δ,max_cg_iter,κ_step,κ_cg)

Approximately solves the quadratic program

`minₛ 1/2 sᵀHs + sᵀg`

` s.t. As = 0`

`xₗ ≤ x + s ≤ xᵤ`

`||s|| ≤ Δ`

by the gradient projection method.

In the QP model, `||.||` denotes the `∞`-norm `||s|| = maxᵢ |sᵢ|`.



First, a Cauchy step ensuring sufficient decrease is computed.

Then, the projected conjugate gradient method is applied to further minimize
the objective in the subspace where the components corresponding to active
bounds of the Cauchy step are fixed.

# Arguments

- `x::Vector`: current iterate
- `g::Vector`: gradient of the Augmented Lagrangian at `x`
- `H::ALHessian`: approximation of the Hessian of the Augmented Lagrangian at `x`
- `xₗ::Vector`: lower bounds on `x`
- `xᵤ::Vector`: upper bounds on `x`
- `Δ::Float64`: trust region radius
- `max_cg_iter::Int`: number of maximum uses of the conjugate gradient method
- `κ_step::Float64`: positive constant used to define the convergence criteria
relative of the gradient projection method
- `κ_cg::Float64`: positve constant used to define the convergence criteria of the
 conjugate gradient method

# On return

- `s::Vector`: Trial step
- `pred::Float64`: Reduction of the quadratic model after taking step `s`.

"""
function projected_gradient(
    x::Vector{T},
    g::Vector{T},
    hess_op::ALHessian{T},
    proj_op::SubspaceProjector{T},
    x_low::Vector{T},
    x_upp::Vector{T},
    radius::T,
    kappa_cg::T) where T

    # Lower bounds on the step
    s_low,s_upp = step_bounds(x,x_low,x_upp,radius)

    # Cauchy step
    s = cauchy_step(s,g,hess_op,proj_op,x_low,x_upp,s_low,s_upp)
    Hs = hess_op*s

    # Apply conjugate gradient if at least one free variable
    saturated_active_set = nb_fixed(proj_op) < nbmax_fixed_bounds(proj_op)

    if !saturated_active_set

        # Prepare for conjugate gradient iterations
        s_low .-= s
        s_upp .-= s
        cg_rhs = Hs .+ g

        # Compute search direction via  projected conjugate gradient
        w, cg_status = pcg(cg_rhs,hess_op,proj_op,s_low,s_upp,kappa_cg)

        # Update the step
        s .+= w
        Hs .= hess_op*s
    end

    pred = dot(g,s) + 0.5*dot(s,Hs)

    return s, pred
end

""" next_breakpoint(d,s,dₗ,dᵤ,fix_bounds)

Finds the smallest scalar `θ` such that one or more components not in `fix_bounds`
of `s + θ*d` lie at one of their bounds `dₗ` or `dᵤ`.

Returns the scalar `θ` and `idx`, the index of the components that becomes active.
"""
function next_breakpoint(
        d::Vector{T},
        s::Vector{T},
        d_l::Vector{T},
        d_u::Vector{T},
        fix_bounds::BitVector;
        atol::T=sqrt(eps(T))) where T

    theta = Inf # current breakpoint value
    idx = []    # list of bounds indicices becoming active at theta

    for i in axes(d,1)
        if !fix_bounds[i]
            if d[i] < -atol
                theta_try = (d_l[i]-s[i]) / d[i]
            elseif d[i] > atol 
                theta_try = (d_u[i]-s[i]) / d[i]
            else theta_try = Inf
            end

            also_bp = abs(theta_try-theta) < atol 
            
            if also_bp
                push!(idx,i)

            elseif !also_bp && theta_try < theta
                theta = theta_try
                idx = [i]
            end
        end
    end
    return theta, idx
end

""" 
    cauchy_step(x,g,H,proj_op,xₗ,xᵤ,dₗ,dᵤ)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model
`q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the
projected gradient step `P(x-t*g) - x` with `P` denoting the projection over
`{v | Av = 0 and max(-Δ,xₗ) ≤ x + v ≤ min(Δ,xᵤ)}`.

This method finds the first local minimum of the quadratic model along the
projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

# Arguments

- `x::Vector`: current iterate
- `g::Vector`: gradient of the augmented Lagrangian at current point
- `H`: Hessian approximation of type `ALHessian` at current point
- `proj_op`: [`SubspaceProjector`](@ref) operator to project the gradient onto
linear constraints
- `x_low::Vector`: lower bounds on the variables `x`
- `x_upp::Vector`: upper bounds on the variables `x`
- `d_low::Vector`: lower bounds on the step
- `d_upp::Vector`: upper bounds on the step

# On return

- `s::Vector`: Cauchy step

""" 
function cauchy_step(
    x::Vector{T},
    g::Vector{T},
    hess_op::ALHessian{T},
    proj_op::SubspaceProjector{T},
    x_low::Vector{T},
    x_upp::Vector{T},
    d_low::Vector{T},
    d_upp::Vector{T}) where T

    (m,n) = size(A)
    max_fixed_bounds = nbmax_fixed_bounds(proj_op)
    
    # Buffers 
    s = zeros(n)                    # accumulated Cauchy step 
    d = Vector{Float64}(undef,n)    # projected search direction 

    # Initial projected steepest direction
    mul!(d,proj_op,-g)

    # Check if they are bounds active at x
    prev_tb = 0
    initial_fixed = initial_active_bounds(x,d,x_low,x_upp)
    
    if !isempty(initial_fixed)
        update_subspace_projector!(proj_op,initial_fixed)
    end

    # Update the projection 
    mul!(d,proj_op,-g)

    # Prepare the first interval 
    tb, idx = next_breakpoint(d,s,d_low,d_upp,proj_op.workspace_mat.fixvars)
    gtd = dot(g,d)
    Hd = hess_op*d
    
    found = false

    while !found && nb_fixed(proj_op) < max_fixed_bounds

        # Compute slope and curvature 
        phi_p = gtd + dot(s,Hd)
        phi_pp = dot(d,Hd)

        # Study the current interval [prev_tb, tb) 
        delta_t = (phi_pp > 0 ? -phi_p / phi_pp : 0.0)
        l_interval = tb - prev_tb

        if phi_p >= 0 
            # local minimum at previous breakpoint
            found = true 
        elseif phi_pp > 0 && delta_t < l_interval    
            # local minimum at t = tb - phi_p / phi_pp
            s .+= delta_t .* d 
            found = true
        else 
            # No local minimum in [prev_tb, tb) 
            # Update accumulated step
            s .+= d .* l_interval
            
            # Compute the projected direction on the next interval
            update_subspace_projector!(proj_op,idx)
            mul!(d,proj_op,-g)

            # Prepare for the next interval
            gtd = dot(g,d)
            Hd .= hess_op*d
            
            prev_tb = tb
            tb, idx = next_breakpoint(d,s,d_low,d_upp,proj_op.workpsace_mat.fixvars)
        end

    end

    return s
end


"""
    norm_reduced_v(v,P)

Computes the norm of the reduced vector `Zᵀv` where `Z` is a null space matrix of the set `{v | Av = 0, vᵢ = 0 for i ∈ fix_vars}`
Typically `v` is the gradient of some objective function and the norm of the reduced gradient is involed to evaluate termination criteria.

# Arguments

- `v`: vector whose norm is computed
- `P`: `SubspaceProjector` operator to compute the projection of `v` onto the nullspace of interest
"""
norm_reduced_v(v::Vector,P::SubspaceProjector) = norm(P*v)

# Criticality measure for the traulls algorithm 
# Norm of the negative gradient on the subspace spanned by active constraints (all equalities + actives bounds)

"""
    criticality_measure(g,Pₓ;p=Inf)

Returns the value of the criticality measure at a point `x`.

Computes the `p`-norm of `Pₓ[g]` where `g` is the gradient of the augmented 
Lagrangian with respect to the primal variables evaluated at `x` and `P` is the
projection operator onto the tangent space

`T(x) = null(A) ∩ {d | dᵢ = 0 for i such that xᵢ = ℓ_i or xᵢ = u_i}}`.

**Arguments**

* `g::Vector`: gradient of the augmented Lagrangian at current point

* `Pₓ::SubspaceProjector`: projection operator onto `T(x)` (see [`SubspaceProjector`](@ref))

* `p`: index of the norm computed (default is `Inf`)

**On return** 

* value of `||Pₓ(g)||ₚ`
"""
function criticality_measure(
    g::Vector{T},
    proj_op::SubspaceProjector{T};
    p::T=T(Inf)) where T
    
    return norm(proj_op*(-g), p)
end

