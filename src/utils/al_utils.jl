# Function and derivatives evaluation for the augmented Lagrangian 

"""
    al_obj(rx, cx, y, mu)

Compute the augmented Lagrangian objective function value.

# Arguments
- `rx::Vector`: Residual vector.
- `cx::Vector`: Constraint violation vector.
- `y::Vector`: Lagrange multiplier vector.
- `mu::Float64`: Penalty parameter.

# Returns
- The value of the augmented Lagrangian objective.
"""
function al_obj(
    rx::Vector,
    cx::Vector,
    y::Vector,
    mu::Float64)  
    
    return 0.5*dot(rx,rx) + dot(y,cx) + 0.5*mu*dot(cx,cx)
end

"""
    al_grad!(rx, cx, y, mu, J, C, g)

Compute the gradient of the augmented Lagrangian objective and store it in `g`.

# Arguments
- `rx::Vector`: Residual vector.
- `cx::Vector`: Constraint violation vector.
- `y::Vector`: Lagrange multiplier vector.
- `mu::Float64`: Penalty parameter.
- `J::Matrix`: Jacobian matrix of the residuals.
- `C::Matrix`: Jacobian matrix of the constraints.
- `g::Vector`: Output vector to store the computed gradient (modified in-place).

# Returns
- `Nothing`: The result is stored in `g`.
"""
function al_grad!(
    rx::Vector,
    cx::Vector,
    y::Vector,
    mu::Float64,
    J::Matrix,
    C::Matrix,
    g::Vector)  

    g .= J'*rx + C'*(y + cx .* mu)
    return
end

"""
    al_grad(rx, cx, y, mu, J, C)

Compute and return the gradient of the augmented Lagrangian objective.

# Arguments
- `rx::Vector`: Residual vector.
- `cx::Vector`: Constraint violation vector.
- `y::Vector`: Lagrange multiplier vector.
- `mu::Float64`: Penalty parameter.
- `J::Matrix`: Jacobian matrix of the residuals.
- `C::Matrix`: Jacobian matrix of the constraints.

# Returns
- `g`: gradient of the Augmented Lagrangian evaluated at `x`
"""
function al_grad(
    rx::Vector,
    cx::Vector,
    y::Vector,
    mu::Float64,
    J::Matrix,
    C::Matrix)  
    
    g = Vector{Float64}(undef,size(J,2))
    al_grad!(rx,cx,y,mu,J,C,g)
    
    return g
end

"""
    al_objgrad!(rx, cx, y, mu, J, C, g)

Compute both the augmented Lagrangian objective value and its gradient, storing the gradient in `g`.

# Arguments
- `rx::Vector`: Residual vector.
- `cx::Vector`: Constraint violation vector.
- `y::Vector`: Lagrange multiplier vector.
- `mu::Float64`: Penalty parameter.
- `J::Matrix`: Jacobian matrix of the residuals.
- `C::Matrix`: Jacobian matrix of the constraints.
- `g::Vector`: Output vector to store the computed gradient (modified in-place).

# Returns
- Value of the augmented Lagrangian objective.
"""
function al_objgrad!(
    rx::Vector,
    cx::Vector,
    y::Vector,
    mu::Float64,
    J::Matrix,
    C::Matrix,
    g::Vector)  

    mx = al_obj(rx,cx,y,mu)
    al_grad!(rx,cx,y,mu,J,C,g)

    return mx
end

"""
    al_objgrad(rx, cx, y, mu, J, C)

Compute and return both the augmented Lagrangian objective value and its gradient.

# Arguments
- `rx::Vector`: Residual vector.
- `cx::Vector`: Constraint violation vector.
- `y::Vector`: Lagrange multiplier vector.
- `mu::Float64`: Penalty parameter.
- `J::Matrix`: Jacobian matrix of the residuals.
- `C::Matrix`: Jacobian matrix of the constraints.

# Returns
- Tuple containing the objective value and the gradient vector.
"""
function al_objgrad(
    rx::Vector,
    cx::Vector,
    y::Vector,
    mu::Float64,
    J::Matrix,
    C::Matrix)  

    mx = al_obj(rx,cx,y,mu)
    g = Vector{Float64}(undef,size(J,2))
    al_grad!(rx,cx,y,mu,J,C,g)

    return mx, g
end


"""
    initial_tolerances(μ,ω₀,η₀,κᵪ,κₑ)

Computes and returns `ω` and `η`, the respective optimality and feasibility tolerances for the first outer iteration of the Augmented Lagrangian algorithm [`boconls`](@ref).

# Arguments 

- `μ::Float64`: intitial penalty paramerer associated to the Augmented Lagrangian function
- `ω₀,η₀,κᵪ,κₑ`: positive constants 
"""
function initial_tolerances(
    mu::Float64,
    omega0::Float64,
    eta0::Float64,
    k_crit::Float64,
    k_feas::Float64) 

    omega = omega0 / (mu^k_crit)
    eta = eta0 / (mu^k_feas)
    return omega, eta
end

"""
    least_squares_multipliers(rx, J, C)

Computes the least-squares multipliers estimates by solving the linear least-squares `minᵥ ||Jᵀrx + Cᵀv||₂` derived from the KKT system.

This problem is solved by the normal equations approach, so matrix `C` must be full rank.

# Arguments 
- `rx::Vector`: residuals evaluated at current point 
- `J::Matrix`: Jacobian of the residuals at current point
- `C::Matrix`: Jacobian of the equality constraints at current

"""
function least_squares_multipliers(rx::Vector, J::Matrix, C::Matrix)  

    gf = J'*rx
    chol_cct = cholesky(C*C')
    v = chol_cct.L \ (-C*gf)
    y = chol_cct.U \ v
    
    return y
end

"""
    first_order_multipliers(y, cx, μ)

Update of the Lagrange multipliers in an Augmented Lagrangian algorithm

Computes and returns the first-order multipliers update `y + μ*cx`.

# Arguments 
- `y::Vector`: vector of Lagrange multipliers 
- `cx::Vector`: equality constraints at current point 
- `mu::Float64`: penalty parameter
"""
function first_order_multipliers(y::Vector, cx::Vector, mu::Float64) 
    return y + mu*cx
end

"""
    first_order_multipliers!(y, cx, μ)

Update of the Lagrange multipliers in an Augmented Lagrangian algorithm (in place version)

Overwrites the vector `y` with the first-order multipliers update `y + μ*cx`.

# Arguments 
- `y::Vector`: vector of Lagrange multipliers 
- `cx::Vector`: equality constraints at current point 
- `mu::Float64`: penalty parameter
"""
function first_order_multipliers!(y::Vector, cx::Vector, mu::Float64)  
    y .+= cx .* mu
    return
end

function step_slack!(
    x::Vector,
    y::Vector,
    cx::Vector,
    mu::Float64,
    n_slack::Int,
    p::Int)
    
    n = size(x,1)
    slack_idx = n - n_slack + 1 : n
    ineq_idx = p - n_slack + 1 : p

    x[slack_idx] .= (t -> max(0,t)).(1/mu * y[ineq_idx] .+ cx[ineq_idx] .+ x[slack_idx])

    return
end

"""
    PrimalDualSolution

Mutable structure gathering the informations about a solution computed by a solver. Each function associated to a solver returns a struct of this type.

# Fields 

- `primal_vars`: Optimal solution of the optimization problem found by the solver
- `lagrange_mults`: Lagrange multipliers associated to the equality constraints at the solution
- `objective`: value of the objective function; for least-squares problems, it is the squared sum of residuals
- `criticality`: value of the criticality measure, i.e. measure of optimality, at the solution
- `feasibility`: norm of the equality constraints at the solution
"""
struct PrimalDualSolution
    primal_vars::Vector
    lagrange_mults::Vector
    objective::Float64
    criticality::Float64
    feasibility::Float64
end


"""
    print_solution(sol;io=stdout)

Formats and prints in `io` (default is the stantard output `stdout`) the fields of a primal-dual solution encoded in `sol`.
"""
function print_solution(sol::PrimalDualSolution;io::IO=stdout)
    println(io, "Squared sum of residuals............................: ", @sprintf("%.6e", sol.objective))
    println(io, "Criticality measure.................................: ", @sprintf("%.6e", sol.criticality))
    println(io, "Feasibility of equality constraints.................: ", @sprintf("%.6e", sol.feasibility))

    println(io, "\nPrimal solution...................................")
    (t -> @printf(io, " %.7e ",t)).(sol.primal_vars)
    println(io, "\n\nLagrange multipliers............................")
    (t -> @printf(io, " %.7e ",t)).(sol.lagrange_mults)

end

