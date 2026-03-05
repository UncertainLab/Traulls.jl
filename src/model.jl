"""
    BoxCnls{T} <: AbstractCnlsModel{T}

 Structure representing a nonlinear least-squares problem of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

`low ≤ x ≤ upp.`

Functions `r`, `h` and `g` are two time continuously differentiable.

**Attributes**

* `res!`: Function evaluating the residuals
* `nleq!`: Function evaluating the nonlinear equality constraints
* `nlineq!`: Function evaluating the nonlinear inequality constraints
* `jac_res!`: Function evaluating the Jacobian of the residuals
* `jac_nleq!`: Function evaluating the Jacobian of the nonlinear equality
constraints
* `jac_nlineq!`: Function evaluating the Jacobian of the nonlinear inequality
constraints
* `x_low`: Lower bounds on the variables
* `x_upp`: Upper bounds on the variables
* `x`: Initial solution
* `n`: Number of variables
* `n_slack`: Number of slack variables
* `m`: Number of residuals
* `p`: Total of nonlinear  constraints (equalities + inequalities)

Nonlinear inequality constraints are converted as equality constraints by
adding slack variables`g(x) - u = 0` with `u ≥ 0`.

Constructors are available for both in-place out out of place functions.

For the in-place versions, evaluation functions must return nothing and have the
signature `f!(x,fx)`, with input `x` and the result being stored in `fx`.

For the out-of-place version, evaluation functions must return an output vector
and have the signature `f(x)`.
"""
mutable struct BoxCnls{T<:Real} <: AbstractCnlsModel{T}
    # In-place evaluation functions
    res!
    nleq!
    nlineq!
    jac_res!
    jac_nleq!
    jac_nlineq!

    # Bounds
    x_low::Vector{T}
    x_upp::Vector{T}

    # Starting point
    x::Vector{T}

    # Dimensions
    n::Int
    n_slack::Int
    m::Int
    p::Int
end

# Constructor with in-place evaluation functions for a model with a mix of
# equalities and inequalities
function BoxCnls!(
    r!,
    h!,
    g!,
    jac_r!,
    jac_h!,
    jac_g!,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    n_var::Int,
    m::Int,
    p_eq::Int,
    p_ineq::Int) where T

    # Add slack variables data
    n_slack = p_ineq
    n = n_var + n_slack
    p = p_eq + p_ineq

    # Adjust bounds and starting point
    x_low = vcat(low, zeros(n_slack))
    x_upp = vcat(upp, fill(Inf,n_slack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x,n_slack)
    g!(x0,u0)
    x_start = vcat(x0,s0)

    return BoxCnls(r!,h!,g!,jac_r!,jac_h!,jac_g!,x_low,x_upp,x_start,n,n_slack,m,p)
end

# Constructor with in-place evaluation functions for a model with only
# equalities or inequalities
function BoxCnls!(
    r!,
    c!,
    jac_r!,
    jac_c!,
    low::Vector{T},
    upp::Vector{T},
    x_start::Vector{T},
    n_var::Int,
    m::Int,
    p::Int,
    only_equalities::Bool) where T

    return if only_equalities
        BoxCnls(r!,c!,nothing,jac_r!,jac_c!,nothing,low,upp,x_start,n_var,0,m,p)

    else begin
        n_slack = p
        n = n_var + n_slack
        x_low = vcat(low, zeros(n_slack))
        x_upp = vcat(upp, fill(Inf,n_slack))
        x0 = vcat(x_start,c(x_start))
        BoxCnls(r!,nothing,c!,jac_r!,nothing,jac_c!,x_low,x_upp,x0,n,n_slack,m,p) end
    end
end

# Constructor with out-of-place evaluation functions for a model with a mix of
# equalities and inequalities
function BoxCnls(
    r,
    h,
    g,
    jac_r,
    jac_h,
    jac_g,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    n_var::Int,
    m::Int,
    p_eq::Int,
    p_ineq::Int) where T

    # In-place versions of evaluation functions
    function r!(x,rx)
        rx[1:m] .= r(x)
        return
    end

    function h!(x,hx)
        hx[1:p_eq] .= h(x)
        return
    end

    function g!(x,gx)
        gx[1:p_ineq] .= g(x)
        return
    end

    function jac_r!(x,Jx)
        Jx[1:m,1:n_var] .= jac_r(x)
        return
    end

    function jac_h!(x,Hx)
        Hx[1:p_eq,1:n_var] .= jac_h(x)
        return
    end

    function jac_g!(x,Gx)
        Gx[1:p_ineq,1:n_var] .= jac_g(x)
        return
    end


    return BoxCnls!(r!,h!,g!,jac_r!,jac_h!,jac_g!,low,upp,x0,n_var,m,p_eq,p_ineq)
end

# Constructor with out-of-place evaluation functions for a model with only
# equalities or inequalities
function BoxCnls(
    r,
    c,
    jac_r,
    jac_c,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    n_var::Int,
    m::Int,
    p::Int,
    only_equalities::Bool)

    # In-place versions of evaluation functions
    function r!(x,rx)
        rx[1:m] .= r(x)
        return
    end

    function c!(x,cx)
        cx[1:p] .= c(x)
        return
    end

    function jac_r!(x,Jx)
        Jx[1:m,1:n_var] .= jac_r(x)
        return
    end

    function jac_c!(x,Cx)
        Hx[1:p,1:n_var] .= jac_c(x)
        return
    end

    return BoxCnls!(r!,c!,jac_r!,jac_c!,low,upp,x0,n_var,m,p,only_equalities)
end


"""
    residuals!(model, x, v)

Evaluates the residuals for the given `model` at input vector `x`, storing the
result in `v`.
"""
function residuals!(model::BoxCnls{T}, x::Vector{T}, v::Vector{T}) where T
    x_var = view(x, 1:model.n)
    model.res!(x_var, v)
    return
end

"""
    residuals(model, x)

Returns the residuals for the given `model` evaluated at input vector `x`.
"""
function residuals(model::BoxCnls{T}, x::Vector{T}) where T
    rx = similar(x, model.m)
    residuals!(model, x, rx)
    return rx
end

"""
    nlconstraints!(model, x, v)

Evaluate the nonlinear constraints for the given `model` at input vector `x`,
storing the result in `v`.
"""
function nlconstraints!(model::BoxCnls{T}, x::Vector{T}, cx::Vector{T}) where T

    n, n_slack, p = model.n, model.n_slack, model.p
    n_var = n - n_slack
    p_eq = p - n_slack

    x_var = view(x,1:n_var)

    # Equality constraints components
    if p_eq > 0
        hx = view(cx,1:p_eq)
        model.nleq!(x_var, hx)
    end

    # Inequality constraints transformed into equalities
    if n_slack > 0
        gxmu = view(cx, p_eq+1:p)
        x_slack = view(x, n_var+1:n)
        model.nlineq!(x_var, gx) # gxmu ← g(x)
        gxmu .-= x_slack         # gxmu ← gxmu - u
    end

    return
end

"""
    nlconstraints(model::BoxCnls, x::Vector)

Returns the nonlinear constraints for the given `model` evaluated at input vector
`x`.
"""
function nlconstraints(model::BoxCnls,x::Vector)
    cx = similar(x, model.p)
    nlconstraints!(model, x, cx)
    return cx
end

"""
    jac_residuals!(model, x, J)

Evaluates the Jacobian of the residuals for the given `model` at input vector `x`,
storing the result in matrix `J`.
"""
function jac_residuals!(model::BoxCnls, x::Vector, J::Matrix)
    n, n_slack, m = model.n, model.n_slack, model.m
    n_var = n - n_slack
    x_var = view(x,1:n_var)

    # Derivatives with respect to decision variables
    Jxvar = view(J, 1:m, 1:n)
    model.jac_res!(x_var, Jxvar)

    # Derivatives with respect to slack variables
    if n_slack > 0
        J[:,n_var+1:end] .= zeros(m,n_slack)
    end

    return
end

"""
    jac_residuals(model, x)

Returns the Jacobian of the residuals for the given `model` evaluated at input vector `x`.
"""
function jac_residuals(model::BoxCnls{T}, x::Vector{T}) where T
    Jx = similar(x ,model.m, model.n)
    jac_residuals!(model, x, Jx)
    return Jx
end

"""
    jac_nlconstraints!(model, x, C)

Evaluate the Jacobian of the nonlinear constraints for the given `model` at input
vector `x`, storing the result in matrix `C`.
"""
function jac_nlconstraints!(model::BoxCnls{T}, x::Vector{T}, C::Matrix{T}) where T

    n, n_slack, p = model.n, model.n_slack, model.p
    n_var = n - n_slack
    p_eq = p-n_slack
    ivar = 1:n_var
    eqrows = 1:p_eq

    x_var = view(x,ivar)

    # Equality constraints derivatives with respect to decision variables
    if p_eq > 0
        Chx = view(C, eqrows, ivar)
        model.jac_nleq!(x_var, Chx)
    end

    # Derivatives with respect to slack variables and inequality constraints
    # components
    if n_slack > 0
        islack = n_var+1:n
        ineqrows = p_eq+1:p

        # Equality constraints derivatives wrt slack variables
        C[eqrows, islack] .= T(0)

        # Inequality constraints components
        Cgx = view(C, ineqrows, ivar)
        model.jac_nlineq!(x_var, Cgx)
        C[ineqrows, islack] .= Diagonal{T}(-I,n_slack)
    end

    return
end

"""
    jac_nlconstraints(model, x)

Returns the Jacobian of the nonlinear constraints for the given `model` at input
vector `x`.
"""
function jac_nlconstraints(model::BoxCnls{T},x::Vector{T}) where T
    Cx = similar(x, model.p, model.n)
    jac_nlconstraints!(model, x, Cx)
    return Cx
end

# Testing
