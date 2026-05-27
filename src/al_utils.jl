# Function and derivatives evaluation for the augmented Lagrangian 

"""
    al_obj(rx, cx, y, mu)

Compute the augmented Lagrangian objective function value.

# Arguments
- `rx`: Residual vector.
- `cx`: Constraint violation vector.
- `y`: Lagrange multiplier vector.
- `mu`: Penalty parameter.

# Returns
- The value of the augmented Lagrangian objective.
"""
function al_obj(
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T},
    mu::T) where T

    return (1/2)*dot(rx, rx) + dot(y, cx) + (1/2)*mu*dot(cx, cx)
end

"""
    al_grad!(rx, cx, y, mu, J, C, g)

Compute the gradient of the augmented Lagrangian objective and store it in `g`.

# Arguments
- `rx::AbstractVector{T}`: Residual vector.
- `cx::AbstractVector{T}`: Constraint violation vector.
- `y::AbstractVector{T}`: Lagrange multiplier vector.
- `mu::T`: Penalty parameter.
- `J::AbstractMatrix`: Jacobian matrix of the residuals.
- `C::AbstractMatrix`: Jacobian matrix of the constraints.
- `g::AbstractVector{T}`: Output vector to store the computed gradient (modified in-place).

# Returns
- `Nothing`: The result is stored in `g`.
"""
function al_grad!(
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T},
    mu::T,
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    g::AbstractVector{T}) where T

    g .= J'*rx + C'*(y + cx .* mu)
    return
end

"""
    al_grad(rx, cx, y, mu, J, C)

Compute and return the gradient of the augmented Lagrangian objective.

# Arguments
- `rx::AbstractVector{T}`: Residual vector.
- `cx::AbstractVector{T}`: Constraint violation vector.
- `y::AbstractVector{T}`: Lagrange multiplier vector.
- `mu::T`: Penalty parameter.
- `J::AbstractMatrix`: Jacobian matrix of the residuals.
- `C::AbstractMatrix`: Jacobian matrix of the constraints.

# Returns
- `g`: gradient of the Augmented Lagrangian evaluated at `x`
"""
function al_grad(
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T},
    mu::T,
    J::AbstractMatrix{T},
    C::AbstractMatrix{T}) where T
    
    g = similar(rx, size(J, 2))
    al_grad!(rx, cx, y, mu, J, C, g)
    
    return g
end

"""
    al_objgrad!(rx, cx, y, mu, J, C, g)

Compute both the augmented Lagrangian objective value and its gradient, storing the gradient in `g`.

# Arguments
- `rx::AbstractVector{T}`: Residual vector.
- `cx::AbstractVector{T}`: Constraint violation vector.
- `y::AbstractVector{T}`: Lagrange multiplier vector.
- `mu::T`: Penalty parameter.
- `J::AbstractMatrix`: Jacobian matrix of the residuals.
- `C::AbstractMatrix`: Jacobian matrix of the constraints.
- `g::AbstractVector{T}`: Output vector to store the computed gradient (modified in-place).

# Returns
- Value of the augmented Lagrangian objective.
"""
function al_objgrad!(
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T},
    mu::T,
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    g::AbstractVector{T}) where T

    mx = al_obj(rx, cx, y, mu)
    al_grad!(rx, cx, y, mu, J, C, g)

    return mx
end

"""
    al_objgrad(rx, cx, y, mu, J, C)

Compute and return both the augmented Lagrangian objective value and its gradient.

# Arguments
- `rx::AbstractVector{T}`: Residual vector.
- `cx::AbstractVector{T}`: Constraint violation vector.
- `y::AbstractVector{T}`: Lagrange multiplier vector.
- `mu::T`: Penalty parameter.
- `J::AbstractMatrix`: Jacobian matrix of the residuals.
- `C::AbstractMatrix`: Jacobian matrix of the constraints.

# Returns
- Tuple containing the objective value and the gradient vector.
"""
function al_objgrad(
    rx::AbstractVector{T},
    cx::AbstractVector{T},
    y::AbstractVector{T},
    mu::T,
    J::AbstractMatrix{T},
    C::AbstractMatrix{T}) where T

    mx = al_obj(rx, cx, y, mu)
    g = AbstractVector{T}(undef,size(J,2))
    al_grad!(rx, cx, y, mu, J, C, g)

    return mx, g
end


"""
    initial_tolerances(μ,ω₀,η₀,κᵪ,κₑ)

Computes and returns `ω` and `η`, the respective optimality and feasibility tolerances for the first outer iteration of the Augmented Lagrangian algorithm [`boconls`](@ref).

# Arguments 

- `μ::T`: intitial penalty paramerer associated to the Augmented Lagrangian function
- `ω₀,η₀,κᵪ,κₑ`: positive constants 
"""
function initial_tolerances(
    mu::T,
    omega0::T,
    eta0::T,
    k_crit::T,
    k_feas::T) where T

    omega = omega0 / (mu^k_crit)
    eta = eta0 / (mu^k_feas)
    return omega, eta
end

"""
    least_squares_multipliers(rx, J, C)

Computes the least-squares multipliers estimates by solving the linear least-squares `minᵥ ||Jᵀrx + Cᵀv||₂` derived from the KKT system.

This problem is solved by the normal equations approach, so matrix `C` must be full rank.

# Arguments 
- `rx::AbstractVector{T}`: residuals evaluated at current point
- `J::AbstractMatrix`: Jacobian of the residuals at current point
- `C::AbstractMatrix`: Jacobian of the equality constraints at current

"""
function least_squares_multipliers(
    rx::AbstractVector{T},
    J::AbstractMatrix{T},
    C::AbstractMatrix{T}) where T

    # TODO: Replace this computation by an iterative solving
    gf = J'*rx
    cct = C*C'
    y = zeros(T, size(C, 1))

    if isposdef(cct)
        chol_cct = cholesky(cct)
        v = chol_cct.L \ (-C*gf)
        y .= chol_cct.U \ v
    end

    return y
end

"""
    first_order_multipliers(y, cx, μ)

Update of the Lagrange multipliers in an Augmented Lagrangian algorithm

Computes and returns the first-order multipliers update `y + μ*cx`.

# Arguments 
- `y::AbstractVector{T}`: vector of Lagrange multipliers
- `cx::AbstractVector{T}`: equality constraints at current point
- `mu::T`: penalty parameter
"""
function first_order_multipliers(y::AbstractVector{T}, cx::AbstractVector{T}, mu::T) where T
    return y + mu*cx
end

"""
    first_order_multipliers!(y, cx, μ)

Update of the Lagrange multipliers in an Augmented Lagrangian algorithm (in place version)

Overwrites the vector `y` with the first-order multipliers update `y + μ*cx`.

# Arguments 
- `y::AbstractVector{T}`: vector of Lagrange multipliers
- `cx::AbstractVector{T}`: equality constraints at current point
- `mu::T`: penalty parameter
"""
function first_order_multipliers!(
    y::AbstractVector{T},
    cx::AbstractVector{T},
    mu::T) where T

    y .+= cx .* mu
    return
end

function step_slack!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    cx::AbstractVector{T},
    mu::T,
    n_slack::Int,
    p::Int) where T
    
    n = size(x,1)
    slack_idx = n - n_slack + 1 : n
    ineq_idx = p - n_slack + 1 : p

    x[slack_idx] .= (t -> max(0,t)).(1/mu * y[ineq_idx] .+ cx[ineq_idx] .+ x[slack_idx])

    return
end
