export CnlsModel, CnlsModel!

"""
    CnlsModel{T} <: AbstractCnlsModel{T}

 Structure representing a nonlinear least-squares problem of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

`Ax = b`

`ℓ ≤ x ≤ u.`

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
* `linmat`: Matrix of the linear equality constraints
* `linrhs`: Right handside of the linear equality constraints
* `xlow`: Lower bounds on the variables
* `xupp`: Upper bounds on the variables
* `n`: Number of variables
* `nslack`: Number of slack variables
* `nres`: Number of residuals
* `ncons`: Total of nonlinear  constraints (equalities + inequalities)
* `nlincons`: Total number of linear equality constraints, i.e. number of rows of `linmat`
* `x`: Initial guess for solution

Nonlinear inequality constraints are converted as equality constraints by
adding slack variables`g(x) - u = 0` with `u ≥ 0`.

Constructors are available for both in-place out out of place functions.

For the in-place versions, evaluation functions must return nothing and have the
signature `f!(fx, x)`, with input `x` and the result being stored in `fx`.

For the out-of-place version, evaluation functions must return an output vector of
appropriate dimension and have the signature `f(x)`.
"""
mutable struct CnlsModel{T<:Real} <: AbstractCnlsModel{T}
    # In-place evaluation functions
    res!
    nleq!
    nlineq!
    jac_res!
    jac_nleq!
    jac_nlineq!

    # Linear constraints
    linmat::AbstractMatrix{T}
    linrhs::AbstractVector{T}
    xlow::AbstractVector{T}
    xupp::AbstractVector{T}

    # Dimensions
    n::Int
    nslack::Int
    nres::Int
    ncons::Int
    nlincons::Int

    # Starting point
    x::Vector{T}

    # Counters
    counters::TraullsCounters
end

# Constructor with in-place evaluation functions for a model with a mix of equalities and
# inequalities
function CnlsModel!(
    r!,
    h!,
    g!,
    jac_r!,
    jac_h!,
    jac_g!,
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    neq::Int,
    nineq::Int) where T

    # Slack variables data
    nslack = nineq
    n = nvar + nslack
    ncons = neq + nineq

    # Adjust linear constraints
    xlow = vcat(low, zeros(nslack))
    xupp = vcat(upp, fill(Inf, nslack))
    nlincons = size(A,1)
    lincons = hcat(A, zeros(nlincons, nslack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x0, nslack)
    g!(u0, x0)
    xstart = vcat(x0, u0)

    return CnlsModel(r!, h!, g!, jac_r!, jac_h!, jac_g!, lincons, b, xlow, xupp, n, nslack,
                     nres, ncons, nlincons, xstart, TraullsCounters())
end

# Constructor with out-of-place evaluation functions for a model with a mix of
# equalities and inequalities
function CnlsModel(
    r,
    h,
    g,
    jac_r,
    jac_h,
    jac_g,
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    neq::Int,
    nineq::Int) where T

    # In-place versions of evaluation functions
    function r!(rx, x)
        rx[1:nres] .= r(x)
        return
    end

    function h!(hx, x)
        hx[1:neq] .= h(x)
        return
    end

    function g!(gx, x)
        gx[1:nineq] .= g(x)
        return
    end

    function jac_r!(Jx, x)
        Jx[1:nres,1:nvar] .= jac_r(x)
        return
    end

    function jac_h!(Hx, x)
        Hx[1:neq,1:nvar] .= jac_h(x)
        return
    end

    function jac_g!(Gx, x)
        Gx[1:nineq,1:nvar] .= jac_g(x)
        return
    end

    return CnlsModel!(r!, h!, g!, jac_r!, jac_h!, jac_g!, A, b, low, upp, x0, nvar, nres,
                      neq, nineq)
end


# Constructor with in-place evaluation functions for a model with only
# equalities or inequalities

function CnlsModel!(
    r!,
    c!,
    jac_r!,
    jac_c!,
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    ::Val{:only_equalities}) where T

    CnlsModel(r!, c!, nothing, jac_r!, jac_c!, nothing, A, b, low, upp, nvar, 0, nres,
              ncons, size(A,1), x0, TraullsCounters())
end

function CnlsModel!(
    r!,
    c!,
    jac_r!,
    jac_c!,
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    ::Val{:only_inequalities}) where T

    # Adjust dimensions for slack variables
    nslack = ncons
    n = nvar + nslack

    # Adjust linear constraints
    xlow = vcat(low, zeros(nslack))
    xupp = vcat(upp, fill(Inf,nslack))
    nlincons = size(A, 1)
    lincons = hcat(A, zeros(nlincons, nslack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x0 ,nslack)
    c!(u0, x0)
    xstart = vcat(x0,u0)

    CnlsModel(r!, nothing, c!, jac_r!, nothing, jac_c!, lincons, b, xlow, xupp, n,
              nslack, nres, ncons, nlincons, xstart, TraullsCounters())
end

function CnlsModel(
    r,
    c,
    jac_r,
    jac_c,
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    eq_or_ineq::ConstraintsType) where T

    # In-place versions of evaluation functions
    function r!(rx, x)
        rx[1:nres] .= r(x)
        return
    end

    function c!(cx, x)
        cx[1:ncons] .= c(x)
        return
    end

    function jac_r!(Jx, x)
        Jx[1:nres,1:nvar] .= jac_r(x)
        return
    end

    function jac_c!(Cx, x)
        Cx[1:ncons,1:nvar] .= jac_c(x)
        return
    end

    CnlsModel!(r!, c!, jac_r!, jac_c!, A, b, low, upp, x0, nvar, nres, ncons, eq_or_ineq)
end

# Constructor with in-place evaluation functions for model with mix equalities and
# inequalities and bounds on the parameters
function CnlsModel!(
    r!,
    h!,
    g!,
    jac_r!,
    jac_h!,
    jac_g!,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    neq::Int,
    nineq::Int) where T

    # Slack variables data
    nslack = nineq
    n = nvar + nslack
    ncons = neq + nineq

    # Adjust linear constraints
    xlow = vcat(low, zeros(nslack))
    xupp = vcat(upp, fill(Inf, nslack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x0, nslack)
    g!(u0, x0)
    xstart = vcat(x0, u0)

    CnlsModel(r!, h!, g!, jac_r!, jac_h!, jac_g!, zeros(T,1,1), zeros(T,1),
               xlow, xupp, n, nslack, nres, ncons, 0, xstart, TraullsCounters())
end

# Constructor with out of place evaluation functions for models with mixed equalities and
# inequalities and bounds on the parameters
function CnlsModel(
    r,
    h,
    g,
    jac_r,
    jac_h,
    jac_g,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    neq::Int,
    nineq::Int) where T


    # In-place versions of evaluation functions
    function r!(rx, x)
        rx[1:nres] .= r(x)
        return
    end

    function h!(hx, x)
        hx[1:neq] .= h(x)
        return
    end

    function g!(gx, x)
        gx[1:nineq] .= g(x)
        return
    end

    function jac_r!(Jx, x)
        Jx[1:nres,1:nvar] .= jac_r(x)
        return
    end

    function jac_h!(Hx, x)
        Hx[1:neq,1:nvar] .= jac_h(x)
        return
    end

    function jac_g!(Gx, x)
        Gx[1:nineq,1:nvar] .= jac_g(x)
        return
    end

    CnlsModel!(r!, h!, g!, jac_r!, jac_h!, jac_g!, low, upp, x0, nvar, nres, neq, nineq)
end

# Constructor with in-place evaluation functions for model with only equalities or
# and bounds on the parameters

function CnlsModel!(
    r!,
    c!,
    jac_r!,
    jac_c!,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    ::Val{:only_equalities}) where T

    CnlsModel(r!, c!, nothing, jac_r!, jac_c!, nothing, zeros(T,1,1), zeros(T,1), low, upp,
              nvar, 0, nres, ncons, 0, x0, TraullsCounters())
end

# Constructor with in-place evaluation functions for model with only inequalities and bounds
# constraints
function CnlsModel!(
    r!,
    c!,
    jac_r!,
    jac_c!,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    ::Val{:only_inequalities}) where T

    # Adjust dimensions for slack variables

    nslack = ncons
    n = nvar + nslack

    # Adjust linear constraints
    xlow = vcat(low, zeros(nslack))
    xupp = vcat(upp, fill(Inf,nslack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x0 ,nslack)
    c!(u0, x0)
    xstart = vcat(x0,u0)

    CnlsModel(r!, nothing, c!, jac_r!, nothing, jac_c!, zeros(T,1,1), zeros(T,1), xlow,
              xupp, n, nslack, nres, ncons, 0, xstart, TraullsCounters())
end

# Constructor for model with only equalities or inequalities and bounds on the variables
function CnlsModel(
    r,
    c,
    jac_r,
    jac_c,
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Int,
    nres::Int,
    ncons::Int,
    eq_or_ineq::ConstraintsType) where T

    # In-place versions of evaluation functions
    function r!(rx, x)
        rx[1:nres] .= r(x)
        return
    end

    function c!(cx, x)
        cx[1:ncons] .= c(x)
        return
    end

    function jac_r!(Jx, x)
        Jx[1:nres,1:nvar] .= jac_r(x)
        return
    end

    function jac_c!(Cx, x)
        Cx[1:ncons,1:nvar] .= jac_c(x)
        return
    end

    CnlsModel!(r!, c!, jac_r!, jac_c!, low, upp, x0, nvar, nres, ncons, eq_or_ineq)
end

"""
    residuals!(model, rx, x)

Evaluates the residuals for the given `model` at input vector `x`, storing the
result in `res`.
"""
function residuals!(model::CnlsModel{T}, rx::Vector{T}, x::Vector{T}) where T
    x_var = view(x, 1:model.n)
    model.res!(rx, x_var)
    model.counters.nres_eval += 1
    return
end

"""
    residuals(model, x)

Returns the residuals for the given `model` evaluated at input vector `x`.
"""
function residuals(model::CnlsModel{T}, x::Vector{T}) where T
    rx = similar(x, model.nres)
    residuals!(model, rx, x)
    return rx
end

"""
    nlconstraints!(model, cx, x)

Evaluate the nonlinear constraints for the given `model` at input vector `x`,
storing the result in `cx`.
"""
function nlconstraints!(model::CnlsModel{T}, cx::Vector{T}, x::Vector{T}) where T

    n, n_slack, p = model.n, model.nslack, model.ncons
    n_var = n - n_slack
    p_eq = p - n_slack

    x_var = view(x,1:n_var)


    # Equality constraints components
    if p_eq > 0
        hx = view(cx,1:p_eq)
        model.nleq!(hx, x_var)
    end

    # Inequality constraints transformed into equalities
    if n_slack > 0
        gxmu = view(cx, p_eq+1:p)    # buffer for g(x) - u
        x_slack = view(x, n_var+1:n)
        model.nlineq!(gxmu, x_var)   # gxmu ← g(x)
        gxmu .-= x_slack             # gxmu ← gxmu - u
    end

    model.counters.ncons_eval += 1

    return
end

"""
    nlconstraints(model, x)

Returns the nonlinear constraints for the given `model` evaluated at input vector
`x`.
"""
function nlconstraints(model::CnlsModel{T}, x::Vector{T}) where T
    cx = similar(x, model.ncons)
    nlconstraints!(model, cx, x)
    return cx
end

"""
    jac_residuals!(model, J, x)

Evaluates the Jacobian of the residuals for the given `model` at input vector `x`,
storing the result in matrix `J`.
"""
function jac_residuals!(model::CnlsModel{T}, J::Matrix{T}, x::Vector{T}) where T
    n, n_slack, m = model.n, model.nslack, model.nres
    n_var = n - n_slack
    x_var = view(x,1:n_var)

    # Derivatives with respect to decision variables
    Jxvar = view(J, 1:m, 1:n_var)
    model.jac_res!(Jxvar, x_var)

    # Derivatives with respect to slack variables
    if n_slack > 0
        J[:,n_var+1:end] .= zeros(m,n_slack)
    end

    model.counters.njacres_eval += 1
    return
end

"""
    jac_residuals(model, x)

Returns the Jacobian of the residuals for the given `model` evaluated at input vector `x`.
"""
function jac_residuals(model::CnlsModel{T}, x::Vector{T}) where T
    Jx = similar(x, model.nres, model.n)
    jac_residuals!(model, Jx, x)
    return Jx
end

"""
    jac_nlconstraints!(model, C, x)

Evaluate the Jacobian of the nonlinear constraints for the given `model` at input
vector `x`, storing the result in matrix `C`.
"""
function jac_nlconstraints!(model::CnlsModel{T}, C::Matrix{T}, x::Vector{T}) where T

    n, n_slack, p = model.n, model.nslack, model.ncons
    n_var = n - n_slack
    p_eq = p - n_slack

    ivar = 1:n_var
    eqrows = 1:p_eq

    x_var = view(x,ivar)

    # Equality constraints derivatives with respect to decision variables
    if p_eq > 0
        Chx = view(C, eqrows, ivar)
        model.jac_nleq!(Chx, x_var)
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
        model.jac_nlineq!(Cgx, x_var)
        C[ineqrows, islack] .= Diagonal{T}(-I,n_slack)
    end

    model.counters.njaccons_eval += 1
    return
end

"""
    jac_nlconstraints(model, x)

Returns the Jacobian of the nonlinear constraints for the given `model` at input
vector `x`.
"""
function jac_nlconstraints(model::CnlsModel{T},x::Vector{T}) where T
    Cx = similar(x, model.ncons, model.n)
    jac_nlconstraints!(model, Cx, x)
    return Cx
end

function print(io::IO, model::CnlsModel)

    n, nslack, nres, ncons = model.n, model.nslack, model.nres, model.ncons
    nvar = n - nslack

    println(io, "Problem dimensions")
    println(io, "Number of parameters.................: ", @sprintf("%5i", nvar))
    println(io, "Number of slack variables............: ", @sprintf("%5i", nslack))
    println(io, "Number of residuals..................: ", @sprintf("%5i", nres))
    println(io, "Number of nonlinear constraints......: ", @sprintf("%5i", ncons))
    println(io, "Number of linear constraints.........: ", @sprintf("%5i", model.nlincons))
    println(io, "Number of lower bounds...............: ", @sprintf("%5i",
                                                                    count(isfinite, model.xlow)))
    println(io, "Number of upper bounds...............: ", @sprintf("%5i",
                                                                    count(isfinite, model.xupp)))
end

println(io::IO, model::CnlsModel) = print(io,"\n", model)
