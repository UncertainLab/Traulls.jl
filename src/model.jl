# Abstract type defined just for testing
abstract type AbstractCnlsModel{T} end

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

    # Dimensions
    n::Int
    n_slack::Int
    m::Int
    p::Int

    # Starting point
    x::Vector{T}

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
    u0 = similar(x0,n_slack)
    g!(x0,u0)
    x_start = vcat(x0,u0)

    return BoxCnls(r!,h!,g!,jac_r!,jac_h!,jac_g!,x_low,x_upp,n,n_slack,m,p,x_start)
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
    x0::Vector{T},
    n_var::Int,
    m::Int,
    p::Int,
    only_equalities::Bool) where T

    return if only_equalities
        BoxCnls(r!,c!,nothing,jac_r!,jac_c!,nothing,low,upp,n_var,0,m,p,x0)

    else begin
        n_slack = p
        n = n_var + n_slack
        x_low = vcat(low, zeros(n_slack))
        x_upp = vcat(upp, fill(Inf,n_slack))

        # Set initial slack variables to g(x₀)
        u0 = similar(x0,n_slack)
        c!(x0,u0)
        x_start = vcat(x0,u0)

        BoxCnls(r!,nothing,c!,jac_r!,nothing,jac_c!,x_low,x_upp,n,n_slack,m,p,x_start)
    end
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
    println("coucou")
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
    only_equalities::Bool) where T

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
        Cx[1:p,1:n_var] .= jac_c(x)
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
        gxmu = view(cx, p_eq+1:p)    # buffer for g(x) - u
        x_slack = view(x, n_var+1:n)
        model.nlineq!(x_var, gxmu)   # gxmu ← g(x)
        gxmu .-= x_slack             # gxmu ← gxmu - u
    end

    return
end

"""
    nlconstraints(model, x)

Returns the nonlinear constraints for the given `model` evaluated at input vector
`x`.
"""
function nlconstraints(model::BoxCnls{T}, x::Vector{T}) where T
    cx = similar(x, model.p)
    nlconstraints!(model, x, cx)
    return cx
end

"""
    jac_residuals!(model, x, J)

Evaluates the Jacobian of the residuals for the given `model` at input vector `x`,
storing the result in matrix `J`.
"""
function jac_residuals!(model::BoxCnls{T}, x::Vector{T}, J::Matrix{T}) where T
    n, n_slack, m = model.n, model.n_slack, model.m
    n_var = n - n_slack
    x_var = view(x,1:n_var)

    # Derivatives with respect to decision variables
    Jxvar = view(J, 1:m, 1:n_var)
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
using Test, LinearAlgebra, ForwardDiff


@testset "Model with in-place evaluation functions and inequality constraints" begin
    # Problem 65 form Hock-Schittkowski
    n = 3
    m = 3
    p = 1

    # Residuals
    r(x) = [x[1]-x[2],
    (x[1]+x[2]-10)/3,
    x[3]-5.0]

    function r!(x,rx)
        rx .= [x[1]-x[2],
               (x[1]+x[2]-10)/3,
               x[3]-5.0]
        return
    end

    jac_r(x) = [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]

    function jac_r!(x,J)
        J .= [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]
        return
    end
    # Equality constraints
    c(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
    jac_c(x) = [-2x[1] -2x[2] -2x[3]]

    function c!(x,cx)
        cx .= [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
        return
    end

    function jac_c!(x,C)
        C .= [-2x[1] -2x[2] -2x[3]]
        return
    end

    # Bounds
    x_low = [-4.5, -4.5, -5.0]
    x_upp = [4.5, 4.5, 5.0]

    x0 = [-5, 5, 0.0]
    x = vcat(x0,c(x0))

    # Testing for model defined with in-place methods
    model = BoxCnls!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,false)

    @test size(x,1) == model.n
    @test residuals(model,x) ≈ r(x0)
    @test model.n_slack == p
    @test jac_residuals(model,x) ≈ [1. -1. 0 0;
        1/3 1/3 0. 0;
        0. 0. 1. 0;]
    @test jac_nlconstraints(model,x) ≈ hcat(jac_c(x0),[-1])
    @test nlconstraints(model,x) ≈ zeros(1)
    @test model.nleq! === nothing && model.jac_nleq! === nothing

    # Testing for model defined with standard functions
    model = BoxCnls(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,false)

    @test size(x,1) == model.n
    @test residuals(model,x) ≈ r(x0)
    @test model.n_slack == p
    @test jac_residuals(model,x) ≈ [1. -1. 0 0;
        1/3 1/3 0. 0;
        0. 0. 1. 0;]
    @test jac_nlconstraints(model,x) ≈ hcat(jac_c(x0),[-1])
    @test nlconstraints(model,x) ≈ zeros(1)
    @test model.nleq! === nothing && model.jac_nleq! === nothing
end

@testset "Model with only equality constraints" begin
    n = 10
    m = 2(n-1)
    p = n-2

    # Residuals

    function r(x)
        n = length(x)
        m = 2(n-1)
        rx = Vector{eltype(x)}(undef,m)
        rx[1:n-1] = [10(x[i]^2 - x[i+1]) for i=1:n-1]
        rx[n:m] = [x[k-n+1] - 1 for k=n:m]
        return rx
    end

    function r!(x,rx)
        n = size(x,1)
        m = 2(n-1)

        rx[1:n-1] .= [10(x[i]^2 - x[i+1]) for i=1:n-1]
        rx[n:m] .= [x[k-n+1] - 1 for k=n:m]
        return
    end

    function jac_r(x)
        n = size(x,1)
        m = 2(n-1)
        J = zeros(eltype(x), (m,n))

        for i=1:n-1
            J[i,i] = 20x[i]
            J[i,i+1] = -10
        end

        for i=n:m
            J[i,i-n+1] = 1
        end
        return J
    end

    function jac_r!(x, J)
        n = size(x,1)
        m = 2(n-1)

        for i=1:n-1
            J[i,i] = 20x[i]
            J[i,i+1] = -10
        end

        for i=n:m
            J[i,i-n+1] = 1
        end
        return
    end

    # Constraints

    function c(x)
        n = length(x)
        cx = [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return cx
    end

    function c!(x, cx)
        n = length(x)
        cx .= [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return cx
    end

    function jac_c(x)
        n = size(x,1)
        A = zeros(eltype(x), (n-2,n))
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return A
    end

    function jac_c!(x,A)
        n = size(x,1)
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return
    end

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)


    # Starting point
    x0 = [(mod(i,2) == 1 ? -1.2 : 1.0) for i=1:n]

    model = BoxCnls!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,true)
    @test size(x0,1) == model.n
    @test model.n_slack == 0
    @test model.p == p
    @test residuals(model,x0) ≈ r(x0)
    @test jac_residuals(model,x0) ≈ jac_r(x0)
    @test jac_nlconstraints(model,x0) ≈ jac_c(x0)
    @test nlconstraints(model,x0) ≈ c(x0)
    @test model.nlineq! === nothing && model.jac_nlineq! === nothing

    # Testing for model defined with standard functions
    model = BoxCnls(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,true)

    @test size(x0,1) == model.n
    @test model.n_slack == 0
    @test model.p == p
    @test residuals(model,x0) ≈ r(x0)
    @test jac_residuals(model,x0) ≈ jac_r(x0)
    @test jac_nlconstraints(model,x0) ≈ jac_c(x0)
    @test nlconstraints(model,x0) ≈ c(x0)
    @test model.nlineq! === nothing && model.jac_nlineq! === nothing
end

@testset "Model with a mix of equalities and inequalies" begin

    n = 5
    m = 4
    p_eq = 1
    p_ineq = 1

    # Residuals

    r(x) = [exp((x[i]-i)^2 / 2x[i+1]^2) for i=1:n-1]

    function r!(x,rx)
        rx .= [exp((x[i]-i)^2 / 2x[i+1]^2) - i for i=1:n-1]
    end

    function jac_r(x)
        n = size(x,1)
        J = zeros(n-1,n)

        for i=1:n-1
            J[i,i] = 2x[i] * exp((x[i]-i)^2 / 2x[i+1]^2)
            J[i,i+1] = exp((x[i]-i)^2 / 2x[i+1]^2) / x[i+1]^3
        end

        return J
    end

    function jac_r!(x,J)
        J .= 0
        for i=1:size(x,1)-1
            J[i,i] = 2x[i] * exp((x[i]-i)^2 / 2x[i+1]^2)
            J[i,i+1] = exp((x[i]-i)^2 / 2x[i+1]^2) / x[i+1]^3
        end
        return
    end

    # Constraints

    g(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

    function g!(x,gx)
        gx .= [1 - x[1]^2 - x[2]^2 - x[3]^2]
        return
    end

    jac_g(x) =  [-2x[1] -2x[2] -2x[3] 0.0 0.0]

    function jac_g!(x,Gx)
        Gx .= [-2x[1] -2x[2] -2x[3] 0.0 0.0]
    end

    h(x) = [x[4]^2 + x[5]^2 - 1]

    function h!(x,hx)
        hx .= [x[4]^2 + x[5]^2 - 1]
        return
    end

    jac_h(x) = [0.0 0.0 0.0 2x[4] 2x[5]]

    function jac_h!(x,Hx)
        Hx .= [0.0 0.0 0.0 2x[4] 2x[5]]
        return
    end

    x_low = [0.1^i for i=1:n]
    x_upp = [Float64(i^2) for i=1:n]

    # Starting point
    x0 = [1/i for i=1:n]

    # Model with in place methods
    model = BoxCnls!(r!,h!,g!,jac_r!,jac_h!,jac_g!,x_low,x_upp,x0,n,m,p_eq,p_ineq)

    @test n+p_ineq == model.n
    @test model.n_slack == 1
    @test model.p == p_eq+p_ineq
    @test size(model.x,1) == model.n
    @test residuals(model,model.x) ≈ r(x0)
    @test jac_residuals(model,model.x) ≈ hcat(jac_r(x0),zeros(m))
    @test nlconstraints(model,model.x) ≈ vcat(h(x0),g(x0)-g(x0))
    @test jac_nlconstraints(model,model.x) ≈ vcat(hcat(jac_h(x0),zeros(1,p_ineq)),
                                                  hcat(jac_g(x0), Matrix{Float64}(-I,p_ineq,p_ineq)))

    # Model with out of place methods
    model = BoxCnls(r,h,g,jac_r,jac_h,jac_g,x_low,x_upp,x0,n,m,p_eq,p_ineq)

    @test n+p_ineq == model.n
    @test model.n_slack == 1
    @test model.p == p_eq+p_ineq
    @test size(model.x,1) == model.n
    @test residuals(model,model.x) ≈ r(x0)
    @test jac_residuals(model,model.x) ≈ hcat(jac_r(x0),zeros(m))
    @test nlconstraints(model,model.x) ≈ vcat(h(x0),g(x0)-g(x0))
    @test jac_nlconstraints(model,model.x) ≈ vcat(hcat(jac_h(x0),zeros(1,p_ineq)),
                                                  hcat(jac_g(x0), Matrix{Float64}(-I,p_ineq,p_ineq)))


end
