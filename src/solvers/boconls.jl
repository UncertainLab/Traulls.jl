# Solver for nonlinear least-squares problems wuht nonlinear equality constraints and bound constraints 

"""
    BoxCnls <: AbstractCnlsModel

 Structure representing a nonlinear least-squares problem subject to nonlinear equality constraints and box constraints

* `res`: Function evaluating the residuals
* `nleq`: Function evaluating the nonlinear equality constraints
* `nlineq`: Function evaluating the nonlinear inequality constraints
* `jac_res`: Function evaluating the Jacobian of the residuals
* `jac_nleq`: Function evaluating the Jacobian of the nonlinear equality constraints
* `jac_nlineq`: Function evaluating the Jacobian of the nonlinear inequality constraints
* `x_low`: Lower bounds on the variables
* `x_upp`: Upper bounds on the variables
* `x`: Initial solution
* `n`: Number of variables
* `n_slack`: Number of slack variables
* `m`: Number of residuals
* `p`: Total of nonlinear  constraints (equalities + inequalities)
"""
mutable struct BoxCnls <: AbstractCnlsModel
    res
    nleq
    nlineq
    jac_res
    jac_nleq
    jac_nlineq
    x_low::Vector
    x_upp::Vector
    x::Vector
    n::Int
    n_slack::Int
    m::Int
    p::Int
end

"""
    BoxCnls(r,h,g,jac_r,jac_h,jac_g,low,upp,n_var,m,p_eq,p_ineq)

Constructor for the [`BoxCnls`](@ref) structure.

Encodes a nonlinear least-squares problems of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

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
* `low`: Lower bounds on the variables
* `upp`: Upper bounds on the variables
* `x_start`: Initial solution
* `n_var`: Number of variables
* `m`: Number of residuals
* `p_eq`: Number of nonlinear equality constraints
* `p_ineq`: Number of nonlinear inequality constraints
"""
function BoxCnls(
    r,
    h,
    g,
    jac_r,
    jac_h,
    jac_g,
    low,
    upp,
    x_start,
    n_var::Int,
    m::Int,
    p_eq::Int,
    p_ineq::Int)

    n_slack = p_ineq
    n = n_var + n_slack
    p = p_eq + p_ineq

    x_low = vcat(low, zeros(n_slack))
    x_upp = vcat(upp, fill(Inf,n_slack))

    return BoxCnls(r,h,g,jac_r,jac_h,jac_g,x_low,x_upp,x_start,n,n_slack,m,p)
end

"""
    BoxCnls(r,c, jac_r,jac_c,low,upp,n_var,m,p,only_equalities))

Constructor for the [`BoxCnls`](@ref) structure.

Encodes a nonlinear least-squares problems of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0 or c(x) ≥ 0`

`low ≤ x ≤ upp.`

Nonlinear inequality constraints are converted as equality constraints by
adding slack variables`c(x) - u = 0` with `u ≥ 0`.
 
**Arguments**

* `r`: Function evaluating the residuals
* `c`: Function evaluating the nonlinear constraints
* `jac_r`: Function evaluating the Jacobian of the residuals
* `jac_c`: Function evaluating the Jacobian of the nonlinear constraints
* `low`: Lower bounds on the variables
* `upp`: Upper bounds on the variables
* `x_start`: Initial values of the variables
* `n_var`: Number of variables
* `m`: Number of residuals
* `p`: Number of nonlinear constraints
* `only_equalities`: Boolean indicating the nature of the nonlinear constraints.
 The latter are treated as equalities if set to `true` and as inequalities if not
"""
function BoxCnls(
    r,
    c,
    jac_r,
    jac_c,
    low,
    upp,
    x_start,
    n_var::Int,
    m::Int,
    p::Int,
    only_equalities::Bool)

    return if only_equalities
        BoxCnls(r,c,nothing,jac_r,jac_c,nothing,low,upp,x_start,n_var,0,m,p)
    
    else begin
        n_slack = p
        n = n_var + n_slack
        x_low = vcat(low, zeros(n_slack))
        x_upp = vcat(upp, fill(Inf,n_slack))
        x0 = vcat(x_start,c(x_start))
        BoxCnls(r,nothing,c,jac_r,nothing,jac_c,x_low,x_upp,x0,n,n_slack,m,p) end
    end
end


#= Methods to evaluate residuals, nonlinear constraints and jacobians of a given model
Methods are implemented in both in place and out of place versions  =#



"""
    residuals!(model::BoxCnls, x::Vector, v::Vector)

Compute the residuals for the given model and input vector `x`, storing the result in `v`.
"""
function residuals!(model::BoxCnls, x::Vector, v::Vector)
    v[:] .= model.res(x[1:model.n-model.n_slack])

    return
end

"""
    residuals(model::BoxCnls, x::Vector) 

Return the residuals for the given model and input vector `x` as a new vector.
"""
function residuals(model::BoxCnls,x::Vector)  
    rx = Vector{eltype(x)}(undef,model.m)
    residuals!(model, x, rx)
    return rx
end

"""
    nlconstraints!(model::BoxCnls, x::Vector, v::Vector) 

Compute the nonlinear constraints for the given model and input vector `x`, storing the result in `v`.
"""
function nlconstraints!(model::BoxCnls, x::Vector, v::Vector) 
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

"""
    nlconstraints(model::BoxCnls, x::Vector) 

Return the nonlinear constraints for the given model and input vector `x` as a new vector.
"""
function nlconstraints(model::BoxCnls,x::Vector)  
    cx = Vector{eltype(x)}(undef,model.p)
    nlconstraints!(model, x, cx)
    return cx
end

"""
    jac_residuals!(model::BoxCnls, x::Vector, J::Matrix) 

Compute the Jacobian of the residuals for the given model and input vector `x`, storing the result in matrix `J`.
"""
function jac_residuals!(model::BoxCnls, x::Vector, J::Matrix) 
    n, n_slack, m = model.n, model.n_slack, model.m
    n_var = n - n_slack

    J[:,1:n_var] .= model.jac_res(x[1:n_var])

    if n_slack > 0
        J[:,n_var+1:end] .= zeros(m,n_slack)
    end
    
    return
end

"""
    jac_residuals(model::BoxCnls, x::Vector) 

Return the Jacobian of the residuals for the given model and input vector `x` as a new matrix.
"""
function jac_residuals(model::BoxCnls, x::Vector)  
    Jx = Matrix{eltype(x)}(undef,model.m, model.n)
    jac_residuals!(model, x, Jx)
    return Jx
end

"""
    jac_nlconstraints!(model::BoxCnls, x::Vector, C::Matrix) 

Compute the Jacobian of the nonlinear constraints for the given model and input vector `x`, 
storing the result in matrix `C`.
"""
function jac_nlconstraints!(model::BoxCnls, x::Vector, C::Matrix) 

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

"""
    jac_nlconstraints(model::BoxCnls, x::Vector) 

Return the Jacobian of the nonlinear constraints for the given model and input vector `x` as a new matrix.
"""
function jac_nlconstraints(model::BoxCnls,x::Vector)  
    Cx = Matrix{eltype(x)}(undef,model.p,model.n)
    jac_nlconstraints!(model, x, Cx)
    return Cx
end

"""
    boconls(model; kwargs...)

Solve a bound-constrained nonlinear least-squares problem with equality constraints of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. c(x) = 0`

`ℓ ≤ x ≤ u,`

with an iterative Augmented Lagrangian method.

Starting from an initial guess `x₀` and an initial estimate of the vector of Lagrange multipliers associated to the equality constraints `y₀`, 
each new iterate `xₖ₊₁` is an approximate solution, with respect to a tolerance `ωₖ > 0`, of the subproblem

`minₓ Lₐ(x,yₖ,μₖ) = 1/2 * r(xₖ)ᵀr(xₖ) + c(xₖ)ᵀ[yₖ + μₖ/2 * c(xₖ)]`²`

`s.t. ℓ ≤ x ≤ u,`

for some penalty parameter `μₖ > 0`, a current estimate of the Lagrange multipliers `yₖ` and using `xₖ` as a starting point.

If the new iterate satisfies `||c(xₖ₊₁)||₂ ≤ ηₖ`, for some `ηₖ > 0`, then the Lagrange multipliers are updated by `yₖ₊₁ = yₖ + μₖc(xₖ)` and the tolerances `ωₖ` and `ηₖ` are tightened.

On the contrary, if xₖ₊₁ fails to satisfies the feasibility inequality, the iterate is unchanged, i.e. `(xₖ₊₁,yₖ₊₁) = (xₖ,yₖ)` and the minimization of the subproblem is restarted with 
a higher penalty parameter `μₖ₊₁ = τμₖ`, with `τ > 1`. The tolerances `ωₖ` and `ηₖ` are still reduced but in a weaker maner.

Subproblems are solved by the gradient projection method (see [`projected_gradient`](@ref)). 

This solver works in double relative precision.

# Arguments

- `r`: Function that computes the residuals
- `J`: Jacobian operator of the residuals
- `c`: Function that computes the equality constraints
- `C`: Jacobian operator of the equality constraints
- `n::Int`: Number of variables
- `m::Int`: Number of residuals
- `p::Int`: Number of equality constraints

## Notes 

Arguments `r`, `J`, `c` and `C` must be functions of a single `Vector` argument of size `n`, say `x`, and return a `Vector` or `Matrix` of appropriate dimensions.

For instance, evaluating the residuals must be done by calling `r(x)` and the output must be a `Vector` of size `m`. 
Similarly, `c(x)` must be of size `p`, `J(x)` of size `m × n` and `C(x)` of size `p × n`.

 

# Keyword Arguments

- `x0::Vector`: Initial guess for the variables (default: `zeros(n)`)
- `x_low::Vector`: Lower bounds for the variables (default: `fill(-Inf, n)`)
- `x_upp::Vector`: Upper bounds for the variables (default: `fill(Inf, n)`)
- `output_file_name`: Name of the output file for logging (default: `""` which makes `stdout` the default output stream)
- `verbose`: Boolean. If set to `true`, execution and iterations detail are printed into the output file (default: false)

## Augmented Lagrangian parameters 

- `mu0::Float64`: Initial penalty parameter (default: `10.0`)
- `tau::Float64`: Increase factor for the penalty parameter (default: `100.0`)
- `omega0::Float64`: Constant to set the initial criticality tolerance (default: `1.0`)
- `eta0::Float64`: Constant to set the initial feasibility tolerance (default: `1.0`)
- `feas_tol::Float64`: Tolerance for feasibility of equality constraints (default: `1e-6`)
- `crit_tol::Float64`: Tolerance for criticality (default: `1e-5`)
- `k_crit::Float64`: Positive constant used to initialize and update the subproblem criticality tolerance in the case of poor improvement of the feasibility (default: `1.0`)
- `k_feas::Float64`: Positive constant used to initialize and update the subproblem feasibility tolerance in the case of poor improvement of the feasibility (default: `0.1`)
- `beta_crit::Float64`: Positive constant used to reduce the subproblem criticality tolerance in the case of good improvement of the feasibility (default: `1.0`)
- `beta_feas::Float64`: Positive constant used to reduce the subproblem feasibility tolerance in the case of good improvement of the feasibility (default: `0.9`)


## Trust region parameters

- `accept_treshold::Float64`: Threshold for accepting a step (default: `0.25`)
- `increase_treshold::Float64`: Threshold for very successful steps in order to extend the trust region (default: `0.75`)
- `decrease_factor::Float64`: Reducing factor of the trust region (default: `0.5`)
- `increase_factor::Float64`: Extension factor of the trust region (default: `2.5`)


## Other solver related constants

- `kappa_step::Float64`: Constant to define the tolerance for the projection gradient method  (default: `0.1`)
- `kappa_cg::Float64`: Constant to define the tolerance for the projected conjugate gradient method (default: `0.1`)
- `max_outer_iter`: Maximum number of outer iterations, i.e. number of minimization of the Augmented Lagrangian (default: `200`)
- `max_inner_iter`: Maximum number of iterations when solving each subproblem with the gradient projection method (default: `100`)
- `max_cg_iter`: Maximum number of conjugate gradient iterations (default: `50`)

# Return

Returns the solution vector and additional information encoded in a [`PrimalDualSolution`](@ref).
"""
function boconls(
    model::BoxCnls;
    mu0::Float64 = 10.0,
    tau::Float64 = 10.0,
    omega0::Float64 = 1.0,
    eta0::Float64 = 1.0,
    feas_tol::Float64 = 1e-6,
    omega_rel::Float64 = 1e-7,
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
    kappa_sos::Float64 = sqrt(eps(Float64)),
    kappa_sml_res::Float64 = 0.1,
    hessian_approx::HessianApprox = gn,
    max_iter::Int = 100,
    max_inner_iter::Int = 100,
    max_cg_iter::Int = 50,
    output_file_name::String="",
    verbose::Bool=false)

    # Sanity check
    @assert (0 < accept_treshold <= increase_treshold < 1) && (0 < decrease_factor < 1 < increase_factor) "Invalid trust region paramaters"

    # Trust Region object 
    tr = TrustRegion(accept_treshold, increase_treshold, decrease_factor, increase_factor, neg_ratio_factor)

    # Prepare output stream to log iteration detail
    output_io = (output_file_name == "" ? stdout : open(output_file_name,"w"))

    # Solve the model 

    y, fx, criticality, feasibility = solve(
        model,
        mu0,
        tau,
        omega0,
        eta0,
        feas_tol,
        omega_rel,
        k_crit,
        k_feas,
        beta_crit,
        beta_feas,
        tr,
        kappa_step,
        kappa_cg,
        kappa_sos,
        kappa_sml_res,
        hessian_approx,
        max_iter,
        max_inner_iter,
        max_cg_iter,
        output_io,
        verbose)

    # Close output stream 
    output_file_name != "" && close(output_io)

    return PrimalDualSolution(model.x, y, fx, criticality, feasibility)
end

"""
    solve(model,args...)

Solve a bound-constrained nonlinear least-squares problem with equality constraints of the form

`minₓ 1/2 * ||r(x)||²`

`s.t. c(x) = 0`

` ` ` ` ` ` `ℓ ≤ x ≤ u,`

encoded in a [`BoxCnls`](@ref) `model` with an iterative Augmented Lagrangian method.

For a more general description of the algorithm, see [`boconls(r,J,c,C,n,m,p; kwargs...)`](@ref).

# Arguments 

- `model::BoxCnls`: Encodes the problem to be solved
- `x0::Vector{Float64}`: Initial guess for the variables 
- `mu0::Float64`: Initial penalty parameter (default: `10.0`)
- `tau::Float64`: Increase factor for the penalty parameter (default: `10.0`)
- `omega0::Float64`: Constant to set the initial criticality tolerance (default: `1.0`)
- `eta0::Float64`: Constant to set the initial feasibility tolerance (default: `1.0`)
- `feas_tol::Float64`: Tolerance for feasibility of equality constraints (default: `1e-6`)
- `crit_tol::Float64`: Tolerance for criticality (default: `1e-5`)
- `k_crit::Float64`: Positive constant used to initialize and update the subproblem criticality tolerance 
in the case of poor improvement of the feasibility (default: `1.0`)
- `k_feas::Float64`: Positive constant used to initialize and update the subproblem feasibility tolerance 
in the case of poor improvement of the feasibility (default: `0.1`)
- `beta_crit::Float64`: Positive constant used to reduce the subproblem criticality tolerance 
in the case of good improvement of the feasibility (default: `1.0`)
- `beta_feas::Float64`: Positive constant used to reduce the subproblem feasibility tolerance 
in the case of good improvement of the feasibility (default: `0.9`)
- `tr::TrustRegion`: Encodes a trust region constraint and its update parameters throughout the algorithm
- `kappa_step::Float64`: Constant to define the tolerance for the projection gradient method  
- `kappa_cg::Float64`: Constant to define the tolerance for the projected conjugate gradient method 
- `max_outer_iter`: Maximum number of outer iterations, i.e. number of minimization of the Augmented Lagrangian 
- `max_inner_iter`: Maximum number of iterations when solving each subproblem with the gradient projection method 
- `max_cg_iter`: Maximum number of conjugate gradient iterations 
- `output_io`: IO stream to print iteration details
- `verbose`: Boolean. If set to `true`, execution and iterations detail are printed into the output file

## Note 

When calling this function, the arguments must be ordered as in the above list.

# Return

The argument `x` is updated in place troughout the execution so when the optimization process stops, 
it corresponds to the solution found by the algorithm.

In addition, the following are also returned: 

- `y::Vector{Float64}`: Vector of Lagrange multipliers at the solution found by the algorithm
- `fx::Float64`: Squared sum of the residuals at the solution
- `pix::Float64`: Value of the criticality measure at the solution
- `feas_measure::Float64`: Norm of the equality constraints vector at the solution

"""
function solve(
    model::BoxCnls,
    mu0::Float64,
    tau::Float64,
    omega0::Float64,
    eta0::Float64,
    feas_tol::Float64,
    omega_rel::Float64,
    k_crit::Float64,
    k_feas::Float64,
    beta_crit::Float64,
    beta_feas::Float64,
    tr::TrustRegion,
    kappa_step::Float64,
    kappa_cg::Float64,
    kappa_sos::Float64,
    kappa_sml_res::Float64,
    hessian_approx::HessianApprox,
    max_outer_iter::Int,
    max_inner_iter::Int,
    max_cg_iter::Int,
    output_io::IO,
    verbose::Bool)  

    
    n, m, p = model.n, model.m, model.p
    x, x_low, x_upp = model.x, model.x_low, model.x_upp
    
    x .= max.(model.x_low, min.(x, model.x_upp)) # Make starting point feasible wrt the bound variable

    verbose && print_boconls_header(n,m,p,x_low,x_upp,omega_rel,feas_tol,tau; io=output_io)
    verbose && print_tr_header(tr;io=output_io)
   
    # Buffers 
    rx = residuals(model, x)
    cx = nlconstraints(model, x)
    J = jac_residuals(model, x)
    C = jac_nlconstraints(model, x)
   
    mu = mu0
    omega, eta = initial_tolerances(mu0, omega0, eta0, k_crit, k_feas)  # Initial tolerances 
    y = least_squares_multipliers(rx, J, C)                            # Initial Lagrange multipliers 

    fx = dot(rx,rx)
    feas_measure = norm(cx,Inf)
    # feas_measure = norm(cx)

    g = al_grad(rx,cx,y,mu,J,C)
    pix = criticality_measure(x,g,x_low,x_upp)
    crit_tol = max(omega_rel, omega_rel*pix)
    first_order_critical = feas_measure <= feas_tol && pix <= crit_tol

    iter = 1

    while !first_order_critical && iter <= max_outer_iter

        verbose && print_outer_iter_header(iter,fx,feas_measure,mu,pix,omega; io=output_io)
        
        verbose && println("\n=== outer iter $iter ===")

    
        pix = solve_subproblem(
            model,
            x,
            x_low, 
            x_upp,
            y,
            mu,
            rx,
            cx,
            J,
            C,
            g,
            tr,
            omega,
            kappa_step,
            kappa_cg,
            kappa_sos,
            kappa_sml_res,
            hessian_approx,
            max_inner_iter,
            max_cg_iter;
            verbose=verbose,
            io=output_io)

        feas_measure = norm(cx,Inf)

        if feas_measure <= eta
            first_order_critical = feas_measure <= feas_tol && pix <= crit_tol
            first_order_multipliers!(y,cx,mu)
            if !first_order_critical
                # Update the iterate, multipliers and decrease tolerances (penalty parameter is unchanged)
                omega = max(omega / mu^beta_crit, omega_rel)
                eta = max(eta / mu^beta_feas, feas_tol)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances (iterate and multipliers are unchanged)
            mu *= tau
            omega = max(omega0 / mu^k_crit, omega_rel)
            eta = max(eta0 / mu^k_feas, feas_tol)
        end

        iter += 1
        fx  = dot(rx,rx)
    end
    
    verbose && print_termination_info(iter,x,y,mu,fx,pix,feas_measure;io=output_io)

    model.x .= x
    return y, fx, pix, feas_measure
end

"""
    solve_subproblem(model, args...)

Solves the outer iteration subproblem 

`minₓ Lₐ(x,y,μ) = 1/2 * r(x)ᵀr(x) + c(x)ᵀ[y + μ/2 * c(x)]`

`s.t. ℓ ≤ x ≤ u,`

using the gradient projection method with trust region. 

The starting point `x₀` and optimality tolerance `ω` are given. The Lagrange multipliers `y` and penalty parameter `μ` are fixed.

At iteration `k`, a quadratic model of the objective function around `xₖ` is formed by

`qₖ(s) = 1/2 sᵀHₖs + sᵀgₖ,`

with `gₖ = ∇ₓLₐ(xₖ,y,μ)` and `Hₖ ≈ ∇²ₓₓ Lₐ(xₖ,y,μ)`.

The step computation consists into approximately solving the quadratic program

`minₛ qₖ(s)`

`s.t. ℓ ≤ xₖ + s ≤ u`

` ` ` ` ` ` `||s|| ≤ Δₖ,`

where `Δₖ` is the trust region radius and `||.||` denotes the `∞`-norm `||x|| = maxᵢ |xᵢ|`. Because `||x|| ≤ Δₖ ⟺ -Δₖ ≤ xᵢ ≤ Δₖ` for all `i`, 
the feasible domain for the step can actually be formulated as the box 
    
`Bₖ = [max(-Δₖe, ℓ-x), min(Δₖe, u-x)]`, with `e = (1,...,1)`.

# Solving the QP

## Cauchy point 

We start by finding the first local minimizer of the model along the projected gradient path 

`s(t) = Pₖ[xₖ - tgₖ] - xₖ` for  `t ≥ 0,` 
    
`Pₖ` denoting the projection over the feasible domain `Bₖ`.
The corresponding scalar defines a Cauchy step that ensures a sufficient reduction of the objective function. This means that taking the Cauchy step at every iteration is enough to 
solve the subproblem. 

## Beyond the Cauchy point 

In order to provide a better reduction, we then apply the conjugate gradient method to the subspace where the components corresponding 
to bounds active at the Cauchy point are fixed.

The resulting `sₖ` step is then accepted or rejected depending on the value of the ratio of the actual reduction over the reduction predicted by the model

`ρ = (Lₐ(xₖ+sₖ,y,μ) - Lₐ(xₖ,y,μ)) / qₖ(sₖ) - qₖ(0)`.

If `ρ ≥ η₁`, where `η₁ ∈ (0,1)` is a given parameter, then the step is accepted and the radius `Δₖ` is eventually increased.
This translates the fact that there is a good agreement between the objective function and the model.

If `ρ < η₁` (poor agreement), the step is rejected and the minimization is restarted with a smaller trust region.

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

where `P` here denotes the projection operator onto the initial feasible box `[ℓ,u]`.
This quantity measures how close a point is from first-order criticality.

# Arguments 

- `model::BoxCnls`: Structure encoding the original constrained nonlinear least-squares problem to be solved 
- `x::Vector`: Starting point the the outer iteration
- `x_low::Vector`: Lower bounds on the variables
- `x_upp::Vector`: Upper bounds on the variables
- `y::Vector`: Current estimation of the Lagrange multipliers
- `mu::Float64`: Penalty parameter
- `rx::Vector`: Residuals evaluated at `x`
- `cx::Vector`: Equality constraints evaluated at `x`
- `J::Matrix`: Jacobian of the residuals evaluated at `x` 
- `C::Matrix`: Jacobian of the equality constraints evaluated at `x`
- `g::Vector`: Gradient of the Augmented Lagrangian at `x`
- `tr::TrustRegion`: Encodes the trust region constraint and associated constants
- `omega_crit::Float64`: Optimality tolerance
- `kappa_step::Float64`: Constant used to define the stopping criteria of the gradient projection method
- `kappa_cg::Float64`: Constant used to define the stopping criteria of the conjugate gradient iterations
- `max_iter::Int`: maximum number of iterations to solve the outer iteration subproblem
- `max_cg_iter::Int`: maximum number of uses of the conjugate gradient method 
- `verbose::Bool=false`: Boolean to log details into a input/output stream
- `io::IO=stdout`: input/output stream (default is `stdout`)
"""
function solve_subproblem(
    model::BoxCnls,
    x::Vector,
    x_low::Vector,
    x_upp::Vector,
    y::Vector,
    mu::Float64,
    rx::Vector,
    cx::Vector,
    J::Matrix,
    C::Matrix,
    g::Vector,
    tr::TrustRegion,
    omega_rel::Float64,
    kappa_step::Float64,
    kappa_cg::Float64,
    kappa_sos::Float64,
    kappa_sml_res::Float64,
    hessian_approx::HessianApprox,
    max_iter::Int,
    max_cg_iter::Int;
    verbose::Bool=false,
    io::IO=stdout) 

    n, n_slack, p = model.n, model.n_slack, model.p
    x_prev, rx_prev, cx_prev = copy(x), copy(rx), copy(cx)

    # Evaluate objective, first derivatives and Hessian of the AL at current point (x,y)
    # residuals!(model,x,rx); nlconstraints!(model,x,cx)
    # jac_residuals!(model,x,J); jac_nlconstraints!(model,x,C)


    alx = al_objgrad!(rx,cx,y,mu,J,C,g)

    H = @match hessian_approx begin
        $gn     => GN(J,C,mu)
        $sr1    => HybridSR1(J,C,mu)
    end

    set_initial_radius!(tr,g)

    pix = criticality_measure(x,g,x_low,x_upp)
    omega_crit = max(omega_rel, omega_rel*pix)
    solved = pix <= omega_crit
    
    short_circuit = false

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
            H,
            x_low,
            x_upp,
            radius,
            max_cg_iter,
            kappa_step,
            kappa_cg)

        # Trial point undistinguishable from current solution or too small radius
        
        short_circuit = check_stalling(s,x,radius)

        if short_circuit continue end

        # Evaluate and analyze the reduction at trial point x+s
        x .+= s
        residuals!(model,x,rx); nlconstraints!(model,x,cx)
        alx = al_obj(rx,cx,y,mu)

        # Step taken on the slack variables, if any
        if n_slack > 0
            slack_idx = n - n_slack + 1 : n
            ineq_idx = p - n_slack + 1 : p
            
            step_slack!(x,y,cx,mu,n_slack,p)
            s[slack_idx] .= x[slack_idx] .- x_prev[slack_idx] .- s[slack_idx] # Adjust the step 
            cx[ineq_idx] .-= s[slack_idx] # Update the constraints involving slack variables without evaluating
            
            # Add reduction of the true objective function after taking second step to pred  
            pred -= alx
            alx = al_obj(rx,cx,y,mu)
            pred += alx

        end

        ratio = step_ratio(alx_prev, alx, pred)
        # verbose && println("[solve_subproblem] ared = $ared, pred = $pred, ratio = $ratio")

        if accept_step(tr,ratio)
            
            # Update the Hessian 

            if hessian_approx == gn # Gauss-Newton case
                jac_residuals!(model,x,J); jac_nlconstraints!(model,x,C) # Implicitly modifies J and C fields in H
                al_grad!(rx,cx,y,mu,J,C,g)
            
            else # Quasi Newton update
                # Form right handside of the structured secant equation
                H.secant_rhs .= -J'*rx .- C'*(y .+ mu.*cx)
                jac_residuals!(model,x,J); jac_nlconstraints!(model,x,C) # Implicitly modifies J and C fields in H
                al_grad!(rx,cx,y,mu,J,C,g)
                H.secant_rhs .+= g
                update_hessian!(H,s,alx_prev,alx,kappa_sos,kappa_sml_res)
            end

            #update_hessian!(H,J,C)
            pix = criticality_measure(x,g,x_low,x_upp)
            

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
        solved = pix <= omega_crit
        iter += 1
    end

    return pix
end

"""
    projected_gradient(x,g,H,xₗ,xᵤ,Δ,max_cg_iter,κₛ,κᵪ)

Approximately solves the quadratic program 

`minₛ 1/2 sᵀHs + sᵀg`

`s.t. xₗ ≤ x + s ≤ u`

` ` ` ` ` ` `||s|| ≤ Δ`

by the gradient projection method.

In the QP model, `||.||` denotes the `∞`-norm `||s|| = maxᵢ |sᵢ|`.

# Arguments 

- `x::Vector`: Current iterate 
- `g::Vector`: Gradient of the Augmented Lagrangian at `x`
- `H::ALHessian`: Approximation of the Hessian of the Augmented Lagrangian at `x`
- `xₗ::Vector`: Lower bounds on `x`
- `xᵤ::Vector`: Upper bounds on `x`
- `Δ::Float64`: Trust region radius
- `max_cg_iter::Int`: Number of maximum uses of the conjugate gradient method 
- `κₛ::Float64`: Positive constant used to define the convergence criteria relative of the gradient projection method
- `κᵪ::Float64`: Positve constant used to define the convergence criteria of the conjugate gradient method

# On return 

- `s::Vector`: Trial step
- `pred::Float64`: Reduction of the quadratic model after taking step `s`. 

"""
function projected_gradient(
    x::Vector,
    g::Vector,
    H::ALHessian,
    x_low::Vector,
    x_upp::Vector,
    radius::Float64,
    max_cg_iter::Int,
    kappa_step::Float64,
    kappa_cg::Float64)  


    n = size(x,1)
    
    s_low, s_upp = step_bounds(x,x_low,x_upp,radius)
    w_low, w_upp = Vector{Float64}(undef,n), Vector{Float64}(undef,n)

    s, fix_vars = cauchy_step(x,g,H,x_low,x_upp,radius)
    Hs = H*s
    b = Hs .+ g
    
    optimal, cg_stop = false, false
    iter = 1
    
    while !optimal && !cg_stop && iter <= max_cg_iter && !all(fix_vars)

        # Lower and upper bounds for the search direction
        w_low .= s_low .- s 
        w_upp .= s_upp .- s

        w, cg_status = pcg(
            b,
            H,
            w_low,
            w_upp,
            fix_vars,
            kappa_cg)

        s .+= w
        Hs .= H*s
        b .= Hs .+ g 

        # Compute norms of reduced gradients ||Zᵀg|| and ||Zᵀ(Hs+g)||
        norm_reduced_g = norm_reduced_v(g, fix_vars)
        norm_reduced_gnext = norm_reduced_v(b, fix_vars)

        # Evaluate termination criteria 
        optimal = norm_reduced_gnext <= kappa_step * norm_reduced_g
        cg_stop = cg_status == negative_curvature

        # Update the set of fixed variables (implicitly updates the null space matrix Z)
        active_bounds!(s, s_low, s_upp, fix_vars)

        iter += 1
    end

    # Predicted reduction of the model taking step s
    pred = dot(g,s) + 0.5*dot(s,Hs)

    return s, pred
end

""" cauchy_step(x,g,H,ℓ,u,Δ)

Compute a Cauchy step that provides a sufficient reduction of the quadratic model `q(s) = <s,Hs> + <g,s>`.

The step is defined by `s_c = s(t_c)` , where `s(t)`, for `t ≥ 0`, is the projected gradient step `P(x-t*g) - x` with `P` denoting the projection over `{v |  max(-Δe,ℓ-x) ≤ v ≤ min(Δe,u-x)}`.

This method finds the first local minimum of the quadratic model along the projected gradient path, i.e. the first local minimum of `t ↦ q(s(t))` on `[0, ∞)`.

Returns the associated Cauchy step `s` and `fix_vars`, a `BitVector` encoding the indices of active bounds at the Cauchy point `x + s`.

Follows the procedure of algorithm 17.3.1 from Trust Regions Methods (Conn, Gould and Toint, SIAM, 2000). 
"""
function cauchy_step(
    x::Vector,
    g::Vector,
    H::ALHessian,
    x_low::Vector,
    x_upp::Vector,
    radius::Float64) 

    n = size(x,1)
    d = Vector{Float64}(undef,n)                    # projected gradient direction
    s = zeros(n)                                    # accumulated Cauchy step
    fix_vars = falses(n)                   # indices of fixed variables

    # Breakpoints values and group indices
    breakpoints, grp_idx  = sort_breakpoints(x,g,x_low,x_upp,radius)
    prev_tb = 0.0
    d .= -g 


    # Handle the case where the first breakpoint is zero
    # Happens when bounds are active at x
    if iszero(breakpoints[1])
        popfirst!(breakpoints)                      # get rid of breakpoint tb = zero 
        first_active_indx = popfirst!(grp_idx)
        fix_vars[first_active_indx] .= true
        fix_vars[setdiff(1:n,first_active_indx)] .= false
        d .= -g .* .!fix_vars
    end


    gtd = dot(g,d)
    Hd = H*d
    
    for (i, tb) in enumerate(breakpoints)
        
        # Compute slope and curvature 
        phi_p = gtd + dot(s,Hd)
        phi_pp = dot(d,Hd)

        # Study the current interval [prev_tb, tb) 
        delta_t = (phi_pp > 0 ? -phi_p / phi_pp : 0.0)
        l_interval = tb - prev_tb

        if phi_p >= 0
            break 
        elseif phi_pp > 0 && delta_t < l_interval    # local minimum at t = tb - phi_p / phi_pp
            s .+= delta_t .* d 
            break
        end

        # No local minimum in [prev_tb, tb)
        # Prepare for the next interval 
        prev_tb = tb
        newly_active = grp_idx[i]
        fix_vars[newly_active] .= true

        s .+= d .* l_interval
        d .= -g .* .!fix_vars
        gtd = dot(g,d)
        Hd = H*d
    end

    
    return s, fix_vars
end

"""
    norm_reduced_v(v,fix_vars)

Computes the norm of the reduced vector `Zᵀv` where `Z` is a null space matrix of the set `{v | vᵢ = 0 for i ∈ fix_vars}`
Typically `v` is the gradient of some objective function and the norm of the reduced gradient is involed to evaluate termination criteria.

# Arguments

- `v`: vector whose norm is computed
- `fix_vars`: `BitVector` encoded the components of `v` that are set to `0`
"""
norm_reduced_v(v::Vector,fix_vars::BitVector) = norm(v[.!fix_vars])


"""
    criticality_measure(x,g,xₗ,xᵤ)

Computes the criticality measure used to measure if a primal-dual solution `(x,y)` is a first-order critical point or not.
    
# Arguments 

- `x::Vector`: Current iterate 
- `g::Vector`: Gradient of the Augmented Lagrangian at current primal-dual iterate `(x,y)` 
- `xₗ::Vector`: Lower bounds on `x`
- `xᵤ::Vector`: Upper bounds on `x`
- `p::Float64`: Nature of the norm computed (default is `Inf`).

# Return 

- `πₓ = ||P[x-g] - x||` where `P` denotes the projection onto the box `[xₗ, xᵤ]` and `||.||` is the `p`-norm for some `p > 1`. 
In practice, either the `ℓ₂` or `∞` norms are used.
 
"""
function criticality_measure(
    x::Vector,
    g::Vector,
    x_low::Vector,
    x_upp::Vector;
    p::Float64=Inf) 

    proj_g = Vector{Float64}(undef,size(x,1))
    project!(proj_g, x .- g, x_low, x_upp)
    pix = norm(proj_g .- x, p)
    
    return pix
end
