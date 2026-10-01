#=
    sparse_model.jl

Structure and methods for constrained nonlinear least-squares models with sparse Jacobians
given by their sparsity patterns.

Author(s): Pierre Borie
=#

export SparseCnlsModel, SparseCnlsModel!

"""
    SparseCnlsModel{Tv,Ti} <: AbstractCnlsModel{Tv}

Structure representing a nonlinear least-squares problem with sparse Jacobians, of the form

`minₓ 1/2 * r(x)ᵀr(x)`

`s.t. h(x) = 0`

`g(x) ≥ 0`

`Ax = b`

`ℓ ≤ x ≤ u.`

Functions `r`, `h` and `g` are twice continuously differentiable and their Jacobians are
sparse matrices, whose sparsity patterns are given in coordinate format, i.e. as the row
and column indices of their nonzero entries. Type parameters `Tv` and `Ti` are,
respectively, the type of the values and the type of the indices.

# Attributes

- `res!`: Function evaluating the residuals
- `nleq!`: Function evaluating the nonlinear equality constraints
- `nlineq!`: Function evaluating the nonlinear inequality constraints
- `jres!`: Function evaluating the Jacobian of the residuals
- `jnleq!`: Function evaluating the Jacobian of the nonlinear equality constraints
- `jnlineq!`: Function evaluating the Jacobian of the nonlinear inequality constraints
- `jr_nzrows`, `jr_nzcols`: Row and column indices of the nonzero entries of the
  Jacobian of the residuals
- `jnleq_nzrows`, `jnleq_nzcols`: Row and column indices of the nonzero entries of the
  Jacobian of the nonlinear equality constraints
- `jnlineq_nzrows`, `jnlineq_nzcols`: Row and column indices of the nonzero entries of
  the Jacobian of the nonlinear inequality constraints
- `linmat`: Matrix of the linear equality constraints
- `linrhs`: Right-hand side of the linear equality constraints
- `xlow`: Lower bounds on the variables
- `xupp`: Upper bounds on the variables
- `n`: Number of variables
- `nslack`: Number of slack variables
- `nres`: Number of residuals
- `ncons`: Total number of nonlinear constraints (equalities + inequalities)
- `nlincons`: Total number of linear equality constraints, i.e. number of rows of `linmat`
- `x`: Initial guess for solution
- `counters`: [`TraullsCounters`](@ref) storing the number of function evaluations

The solver converts the nonlinear inequality constraints into equalities by adding slack
variables. Linear constraints must be provided as equalities.

Constructors are only available for in-place functions. Evaluation functions must return
nothing and have the signature `f!(fx, x)`, with input `x` and the result being stored in
`fx`. Jacobian functions have the same signature `jac!(Jx, x)`, where `Jx` is a sparse
matrix that already has the sparsity pattern given to the constructor. They must only
fill the entries of this pattern, so make sure that they respect it.

The default constructor method has signature

`SparseCnlsModel!(r!, h!, g!, jac_r!, jac_h!, jac_g!, jr_nzrows, jr_nzcols, jh_nzrows,
jh_nzcols, jg_nzrows, jg_nzcols, A, b, ℓ, u, x0, nvar, nres, neq, nineq)`

and requires

- `r!`, `h!`, `g!`: evaluation functions of the residuals `r` and constraints `h`, `g`
- `jac_r!`, `jac_h!`, `jac_g!`: evaluation functions of the respective Jacobians of `r`,
  `h`, `g`
- `jr_nzrows`, `jr_nzcols`: row and column indices of the nonzero entries of the Jacobian
  of `r`
- `jh_nzrows`, `jh_nzcols`: row and column indices of the nonzero entries of the Jacobian
  of `h`
- `jg_nzrows`, `jg_nzcols`: row and column indices of the nonzero entries of the Jacobian
  of `g`
- `A`, `b`: matrix coefficients and right-hand side of the linear equality constraints
- `ℓ`, `u`: bounds on the decision variables (set components of unbounded variables to
  `±Inf`)
- `x0`: initial guess for the decision variables
- `nvar`: the number of decision variables
- `nres`: the number of residuals
- `neq`: the number of nonlinear equality constraints
- `nineq`: the number of nonlinear inequality constraints

The row and column index vectors of a same sparsity pattern must have the same length,
otherwise an `ArgumentError` is thrown.

The following methods are available variants of this constructor for problems with only
nonlinear equality constraints, with or without linear equality constraints.

`SparseCnlsModel!(r!, c!, jac_r!, jac_c!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols, A, b,
ℓ, u, x0, nvar, nres, ncons, Val(:only_equalities))`

`SparseCnlsModel!(r!, c!, jac_r!, jac_c!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols, ℓ, u,
x0, nvar, nres, ncons, Val(:only_equalities))`
"""
mutable struct SparseCnlsModel{Tv<:Real, Ti <: Int} <: AbstractCnlsModel{Tv}
    # In-place evaluation functions
    res!
    nleq!
    nlineq!
    jres!
    jnleq!
    jnlineq!

    # Sparsity pattern of jacobian
    jr_nzrows::Vector{Ti}
    jr_nzcols::Vector{Ti}
    jnleq_nzrows::Vector{Ti}
    jnleq_nzcols::Vector{Ti}
    jnlineq_nzrows::Vector{Ti}
    jnlineq_nzcols::Vector{Ti}

    # Linear constraints
    linmat::AbstractMatrix{Tv}
    linrhs::AbstractVector{Tv}
    xlow::AbstractVector{Tv}
    xupp::AbstractVector{Tv}

    # Dimensions
    n::Ti
    nslack::Ti
    nres::Ti
    ncons::Ti
    nlincons::Ti

    # Starting point
    x::Vector{Tv}

    # Counters
    counters::TraullsCounters
end

# Constructor with in-place evaluation functions for a model with a mix of equalities and
# inequalities
function SparseCnlsModel!(
    r!,
    h!,
    g!,
    jr!,
    jh!,
    jg!,
    jr_nzrows::Vector{Ti},
    jr_nzcols::Vector{Ti},
    jh_nzrows::Vector{Ti},
    jh_nzcols::Vector{Ti},
    jg_nzrows::Vector{Ti},
    jg_nzcols::Vector{Ti},
    A::Matrix{T},
    b::Vector{T},
    low::Vector{T},
    upp::Vector{T},
    x0::Vector{T},
    nvar::Ti,
    nres::Ti,
    neq::Ti,
    nineq::Ti) where {T, Ti}

    # Check the sparsity patterns are valid
    spjr_valid = size(jr_nzrows, 1) == (size(jr_nzcols, 1))
    spjh_valid = size(jh_nzrows, 1) == (size(jh_nzcols, 1))
    spjg_valid = size(jg_nzrows, 1) == (size(jg_nzcols, 1))

    !(spjr_valid && spjh_valid && spjg_valid) &&
        throw(ArgumentError("Incoherent sparsity pattern"))

    # Slack variables data
    nslack = nineq
    n = nvar + nslack
    ncons = neq + nineq

    # Adjust linear constraints
    xlow = vcat(low, zeros(T, nslack))
    xupp = vcat(upp, fill(Inf, nslack))
    nlincons = size(A,1)
    lincons = hcat(A, zeros(nlincons, nslack))

    # Set initial slack variables to g(x₀)
    u0 = similar(x0, nslack)
    g!(u0, x0)
    xstart = vcat(x0, u0)

    return SparseCnlsModel(r!, h!, g!, jr!, jh!, jg!, jr_nzrows, jr_nzcols, jh_nzrows,
                           jh_nzcols, jg_nzrows, jg_nzcols, lincons, b, xlow,
                           xupp, n, nslack, nres, ncons, nlincons, xstart,
                           TraullsCounters())
end

# Constructor with in-place evaluation functions for a model where the nonlinear constraints
# are equalities
function SparseCnlsModel!(
    r!,
    c!,
    jr!,
    jc!,
    jr_nzrows::Vector{Ti},
    jr_nzcols::Vector{Ti},
    jc_nzrows::Vector{Ti},
    jc_nzcols::Vector{Ti},
    A::Matrix{T},
    b::Vector{T},
    xlow::Vector{T},
    xupp::Vector{T},
    x0::Vector{T},
    nvar::Ti,
    nres::Ti,
    ncons::Ti,
    ::Val{:only_equalities}) where {T, Ti}

    empty_intvec = Vector{Ti}([])

    SparseCnlsModel(r!, c!, nothing, jr!, jc!, nothing, jr_nzrows, jr_nzcols, jc_nzrows,
                    jc_nzcols, empty_intvec, empty_intvec, A, b, xlow, xupp, nvar, nslack, nres,
                    ncons, size(A, 1), x0, TraullsCounters())
end

# Constructor with in-place evaluation functions for a model where the nonlinear constraints
# are equalities and the linear constraints are bounds on the variables
function SparseCnlsModel!(
    r!,
    c!,
    jr!,
    jc!,
    jr_nzrows::Vector{Ti},
    jr_nzcols::Vector{Ti},
    jc_nzrows::Vector{Ti},
    jc_nzcols::Vector{Ti},
    xlow::Vector{T},
    xupp::Vector{T},
    x0::Vector{T},
    nvar::Ti,
    nres::Ti,
    ncons::Ti,
    ::Val{:only_equalities}) where {T, Ti}

    empty_intvec = Vector{Ti}([])

    SparseCnlsModel(r!, c!, nothing, jr!, jc!, nothing, jr_nzrows, jr_nzcols, jc_nzrows,
                    jc_nzcols, empty_intvec, empty_intvec, zeros(T, 1, 1), zeros(T, 1), xlow,
                    xupp, nvar, 0, nres, ncons, 0, x0, TraullsCounters())
end

# Sparse versions of the evaluations functions 

# Out-of-place residuals
function residuals!(
    model::SparseCnlsModel{T, Ti},
    rx::AbstractVector{T},
    x::AbstractVector{T}) where {T, Ti}

    x_var = view(x, 1:model.n)
    model.res!(rx, x_var)
    model.counters.nres_eval += 1
    return
end

# In-place residuals
function residuals(
    model::SparseCnlsModel{T, Ti},
    x::AbstractVector{T}) where {T, Ti}

    rx = similar(x, model.nres)
    residuals!(model, rx, x)
    return rx
end

# In-place nonlinear constraints 
function nlconstraints!(model::SparseCnlsModel{T, Ti},
                        cx::AbstractVector{T},
                        x::AbstractVector{T}) where {T, Ti}

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

# Out-of-place nonlinear constraints
function nlconstraints(
    model::SparseCnlsModel{T, Ti},
    x::AbstractVector{T}) where {T, Ti}

    cx = similar(x, model.ncons)
    nlconstraints!(model, cx, x)
    return cx
end

# In-place residuals jacobian
function jac_residuals!(
    model::SparseCnlsModel{T, Ti},
    J::AbstractSparseMatrix{T, Ti},
    x::AbstractVector{T}) where {T, Ti}

    n, n_slack, m = model.n, model.nslack, model.nres
    n_var = n - n_slack
    x_var = view(x, 1:n_var)

    # Derivatives with respect to decision variables
    Jxvar = view(J, 1:m, 1:n_var)
    model.jres!(Jxvar, x_var)

    model.counters.njacres_eval += 1
    return
end

# Out-of-place residuals jacobian 
function jac_residuals(
    model::SparseCnlsModel{T, Ti},
    x::AbstractVector{T}) where {T, Ti}

    # Allocate matrix with appropriate sparsity pattern
    nnz = size(model.jr_nzrows, 1)
    Jx = sparse(model.jr_nzrows, model.jr_nzcols, zeros(T, nnz), model.nres, model.n)
    jac_residuals!(model, Jx, x)
    return Jx
end

# In-place constraints jacobian
function jac_nlconstraints!(model::SparseCnlsModel{T, Ti},
                            C::AbstractSparseMatrix{T, Ti},
                            x::AbstractVector{T}) where {T, Ti}

    n, n_slack, p = model.n, model.nslack, model.ncons
    n_var = n - n_slack
    p_eq = p - n_slack

    ivar = 1:n_var
    eqrows = 1:p_eq

    x_var = view(x, ivar)

    # Equality constraints derivatives with respect to decision variables
    if p_eq > 0
        Chx = view(C, eqrows, ivar)
        model.jnleq!(Chx, x_var)
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
        model.jnlineq!(Cgx, x_var)
    end

    model.counters.njaccons_eval += 1
    return
end

# Out-of-place constraints jacobian
function jac_nlconstraints(model::SparseCnlsModel{T, Ti},
                           x::AbstractVector{T}) where {T, Ti}

    n, n_slack, p = model.n, model.nslack, model.ncons
    n_var = n - n_slack
    neq = p - n_slack

    nnz_nleq = size(model.jnleq_nzrows, 1)
    nnz_nlineq = size(model.jnlineq_nzrows, 1)
    nnz = nnz_nleq + nnz_nlineq + n_slack

    # Form sparsity pattern of the resulting Jacobian matrix
    jc_nzrows = Vector{Ti}([])
    jc_nzcols = Vector{Ti}([])
    nzvalues = zeros(T, nnz)

    # Nonlinear equalities block
    if neq > 0
        append!(jc_nzrows, model.jnleq_nzrows)
        append!(jc_nzcols, model.jnleq_nzcols)
    end


    if n_slack > 0
        # Nonlinear inequalities block
        append!(jc_nzrows, model.jnlineq_nzrows)
        append!(jc_nzcols, model.jnlineq_nzcols)

        # Slack variables block, i.e. -Identity of dimension `nslack`
        append!(jc_nzrows, collect(n_eq+1 : p))
        append!(jc_nzcols, collect(n_var+1 : n))
        mone = T(-1)
        nzvalues[end-n_slack+1:end] .= mone
    end

    Cx = sparse(jc_nzrows, jc_nzcols, nzvalues, p, n)
    jac_nlconstraints!(model, Cx, x)
    return Cx
end

""" 
    print(io::IO, model::SparseCnlsModel)

Overloaded `print` method for a `SparseCnlsModel`.
"""
function print(io::IO, model::SparseCnlsModel)

    n, nslack, nres, ncons = model.n, model.nslack, model.nres, model.ncons
    nvar = n - nslack
    nz_jres = size(model.jr_nzrows, 1)
    nz_jcons = size(model.jnleq_nzrows, 1) + size(model.jnlineq_nzrows, 1)

    println(io, "Problem dimensions")
    println(io, "Number of parameters.......................: ", @sprintf("%5i", nvar))
    println(io, "Number of slack variables..................: ", @sprintf("%5i", nslack))
    println(io, "Number of residuals........................: ", @sprintf("%5i", nres))
    println(io, "Number of nonlinear constraints............: ", @sprintf("%5i", ncons))
    println(io, "Number of non zeros in residuals jacobian..: ", @sprintf("%5i", nz_jres))
    println(io, "Number of non zeros in constraints jacobian: ", @sprintf("%5i", nz_jcons))
    println(io, "Number of linear constraints...............: ", @sprintf("%5i", model.nlincons))
    println(io, "Number of lower bounds.....................: ", @sprintf("%5i",
                                                                    count(isfinite, model.xlow)))
    println(io, "Number of upper bounds.....................: ", @sprintf("%5i",
                                                                    count(isfinite, model.xupp)))
end

println(io::IO, model::SparseCnlsModel) = print(io,"\n", model)
