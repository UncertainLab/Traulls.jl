"""
    SubspaceProjector{T,MT}

Operator computing the orthogonal projection onto a subspace of the form

`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`

where `A` is a full row rank `m × n` (`m < n`) matrix and `fixvars` is a subset
of `{1,…,n}` encoding currently active bounds.

Writing `A₊ = [A ; Z]`, with `Z` the selection rows of the identity matrix
associated to `fixvars`, the projection of `x` is

`P x = x - A₊ᵀ(A₊A₊ᵀ)⁻¹A₊x`.

The factor `L` of the Cholesky decomposition `A₊A₊ᵀ = LLᵀ` is maintained
**incrementally**: activating a variable appends one row/column to `A₊A₊ᵀ` and is
applied as a bordered Cholesky update in `O((m+p)²)` rather than recomputing the
whole factorization. Because the augmented dimension `m + p` never exceeds `n`
(the projected subspace must keep a positive dimension), the factor is stored in
a single preallocated `n × n` buffer and the rows `1:m` (the constant factor of
`AAᵀ`) are never overwritten — so resetting the active set is `O(1)`.

** Attributes

* `A`: equality constraints matrix (`m × n`)
* `fixvars`: `BitVector` of length `n`, `fixvars[i] = true` ⟺ a bound on variable
  `i` is active
* `fixidx`: indices of the active variables, in the order they were activated.
  `fixidx[k]` is the variable represented by augmented row `m + k`. The ordering
  must match the order in which rows were appended to `L`.
* `L`: preallocated `n × n` matrix whose leading `naug × naug` block holds the
  lower Cholesky factor of the current `A₊A₊ᵀ`
* `naug`: current augmented dimension `m + p`
* `m`, `n`: dimensions of `A`
* `aug`, `col`: length-`n` scratch buffers (projection right-hand side and
  incremental-update column), allocated once
"""
mutable struct SubspaceProjector{T<:Real, MT<:AbstractMatrix{T}} <: Projector{T}
    A::MT
    fixvars::BitVector
    fixidx::Vector{Int}
    L::Matrix{T}
    naug::Int
    m::Int
    n::Int
    aug::Vector{T}
    col::Vector{T}
end

"""
    SubspaceProjector(A, chol_AAᵀ)

Constructor for the projection operator onto the null space of `A` (no bound
active). `chol_AAᵀ` is the Cholesky decomposition of `AAᵀ`.
"""
function SubspaceProjector(A::AbstractMatrix{T}, chol_aat::Cholesky{T,Matrix{T}}) where T
    m, n = size(A)
    m >= n && error("DimsError: the equality matrix must have strictly less rows than columns")

    L = zeros(T, n, n)
    @views L[1:m, 1:m] .= chol_aat.L   # broadcasting a LowerTriangular zeroes the upper part

    SubspaceProjector{T, typeof(A)}(A, falses(n), Int[], L, m, m, n,
                                    Vector{T}(undef, n), Vector{T}(undef, n))
end

"""
    SubspaceProjector(A, fixvars, chol_AAᵀ)

Constructor for the projection operator onto
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`. The initial active set encoded in the
`BitVector` `fixvars` is applied through incremental updates.
"""
function SubspaceProjector(
    A::AbstractMatrix{T},
    fixvars::BitVector,
    chol_aat::Cholesky{T,Matrix{T}}) where T

    P = SubspaceProjector(A, chol_aat)
    @inbounds for i in eachindex(fixvars)
        if fixvars[i]
            P.fixvars[i] = true
            append_active!(P, i)
        end
    end
    return P
end

# Bordered Cholesky update: append the row `eⱼᵀ` to `A₊` (variable `j` becomes
# fixed) and update the Cholesky factor `L` of `A₊A₊ᵀ` in place.
#
# The appended column of `A₊A₊ᵀ` is `c = [A[:,j] ; 0_p]` (the `0_p` block comes
# from the orthogonality of distinct selection rows) and the new diagonal entry
# is `eⱼᵀeⱼ = 1`. Solving `L w = c` gives the new row `[wᵀ, √(1 - wᵀw)]`.
#
# Assumes variable `j` is not already represented in rows `m+1:naug` and that
# `naug < n` (guaranteed by the `saturated_subspace` guards in the callers).
function append_active!(P::SubspaceProjector{T}, j::Int) where T
    na = P.naug
    c = view(P.col, 1:na)

    @inbounds for k in 1:P.m
        c[k] = P.A[k, j]
    end
    @inbounds for k in P.m+1:na
        c[k] = zero(T)
    end

    # Forward substitution L w = c, result stored back in c
    ldiv!(LowerTriangular(view(P.L, 1:na, 1:na)), c)

    d = sqrt(one(T) - dot(c, c))

    @inbounds for k in 1:na
        P.L[na+1, k] = c[k]
    end
    P.L[na+1, na+1] = d

    P.naug = na + 1
    push!(P.fixidx, j)
    return
end

# Recompute the factor for the current active set from scratch (used after
# freeing variables, which would otherwise require a Cholesky downdate). The
# remaining active variables are re-appended in their original relative order so
# the factor stays consistent with `fixidx`. Rare path.
function rebuild_factor!(P::SubspaceProjector)
    remaining = filter(j -> P.fixvars[j], P.fixidx)
    P.naug = P.m
    empty!(P.fixidx)
    for j in remaining
        append_active!(P, j)
    end
    return
end

"""
    mul!(r, P::SubspaceProjector, x)

Compute the projection `r = P x = x - A₊ᵀ(A₊A₊ᵀ)⁻¹A₊x` in place, without
allocation. The result is stored in `r`; `r` and `x` must not alias.
"""
function mul!(r::AbstractVector{T}, P::SubspaceProjector{T}, x::AbstractVector{T}) where T
    m, na = P.m, P.naug
    p = na - m
    ab = view(P.aug, 1:na)

    # ab ← A₊x = [A x ; x[fixidx]]
    mul!(view(ab, 1:m), P.A, x)
    @inbounds for k in 1:p
        ab[m+k] = x[P.fixidx[k]]
    end

    # ab ← (A₊A₊ᵀ)⁻¹ A₊x  via the two triangular solves of L Lᵀ
    Lv = LowerTriangular(view(P.L, 1:na, 1:na))
    ldiv!(Lv, ab)
    ldiv!(Lv', ab)

    # r ← x - A₊ᵀ ab
    copyto!(r, x)
    mul!(r, P.A', view(ab, 1:m), -one(T), one(T))   # r -= Aᵀ ab₁
    @inbounds for k in 1:p
        r[P.fixidx[k]] -= ab[m+k]                    # r -= Zᵀ ab₂
    end

    return r
end

"""
    Base.:*(P::SubspaceProjector, x)

Allocating version of the projection: returns `P x`.
"""
function Base.:*(P::SubspaceProjector{T}, x::AbstractVector{T}) where T
    r = Vector{T}(undef, P.n)
    mul!(r, P, x)
    return r
end

# Structure encoding a coordinate subspace where components of vectors
# corresponding to active bounds are set to 0
mutable struct CoordinateSubspaceProjector{T<:Real} <: Projector{T}
    fixvars::BitVector
end

# Constructor for `CoordinateSubspaceProjector` structure.
# Returns a structure with `fixvars` attribute initalized to `falses(n)` where
# `n` is an integer given as input.
# This corresponds to define the underlying subspace to `ℝⁿ`.

CoordinateSubspaceProjector(n::Int;T::DataType=Float64) = CoordinateSubspaceProjector{T}(falses(n))

# Overload the `LinearAlgebra.mul!` method to compute projection of a vector `v`
# onto a coordinate subspace represented by `P` as a matrix-vector product.
# The result is stored in vector `r`.

function mul!(r::Vector, P::CoordinateSubspaceProjector, v::Vector)

    freevars = .!(P.fixvars)

    r[P.fixvars] .= 0          # set rᵢ = 0 for fixed components
    r[freevars] .= v[freevars] # set rᵢ = vᵢ for free components

    return
end

# Overload the `Base.*` method to compute projection of a vector `v`
# onto a coordinate subspace represented by `P` as a matrix-vector product.
# The result is stored in vector `r`.

function Base.:*(P::CoordinateSubspaceProjector, v::Vector)

    res = Vector{eltype(v)}(undef,size(v,1))
    mul!(res,P,v)

    return res
end


"""
    set_active!(proj_op, newly_active)

Activate the bound constraints `vᵢ = 0` for `i ∈ newly_active` in the subspace
encoded by `proj_op`, updating the Cholesky factor incrementally (one bordered
update per newly fixed variable). Already-active variables are skipped.
"""
function set_active!(P::SubspaceProjector, newly_active)
    @inbounds for j in newly_active
        if !P.fixvars[j]
            P.fixvars[j] = true
            append_active!(P, j)
        end
    end
    return
end

# Set component at index `i` active
@inline function set_active!(P::CoordinateSubspaceProjector, i::Int)
    P.fixvars[i] = true
end

# Set active components of indices in `newly_fixed`
@inline function set_active!(P::CoordinateSubspaceProjector, newly_fixed::Vector{Int})
    P.fixvars[newly_fixed] .= true
end


"""
    set_free!(proj_op, freevars)

Deactivate the bound constraints `vᵢ = 0` for `i ∈ freevars` in the subspace
encoded by `proj_op`. Freeing a variable is a Cholesky downdate, so the factor
is rebuilt for the remaining active set (rare path).
"""
function set_free!(P::SubspaceProjector, freevars)
    any_freed = false
    @inbounds for j in freevars
        if P.fixvars[j]
            P.fixvars[j] = false
            any_freed = true
        end
    end
    any_freed && rebuild_factor!(P)
    return
end

# Set free component at index `i`
@inline function set_free!(P::CoordinateSubspaceProjector, i::Int)
    P.fixvars[i] = false
end

# Set free components at indices in `freed`
@inline function set_free!(P::CoordinateSubspaceProjector, freed::Vector{Int})
    P.fixvars[freed] .= false
end


# Returns the number of degrees of freedom remaining into the restricted
# supspace represented by operator `proj_op`

nb_degrees_of_freedom(P::SubspaceProjector) = P.n - P.naug

# Returns the number of degrees of freedoms remaining in the coordinate subspace
# represented by operator `P`.

function nb_degrees_of_freedom(P::CoordinateSubspaceProjector)
    fixed = P.fixvars
    return size(fixed,1) - count(fixed)
end

# Returns `true` if they are no remaining free variables in the subspace represente by the
# projector operator 'P', `false` instead.
saturated_subspace(P::Projector) = nb_degrees_of_freedom(P) == 0

# Returns `true` if the variable at index `i` is fixed in the subspace represented by `proj_op`
is_fixed(P::SubspaceProjector, i::Int) = P.fixvars[i]

# Returns `true` if variable at index `i` is fixed, false if not
is_fixed(P::CoordinateSubspaceProjector, i::Int) = P.fixvars[i]


# Reset the projector operator by setting all bounds as inactive.
# Because incremental updates only ever touch rows `m+1:naug`, restoring the
# `AAᵀ` factor amounts to dropping the appended rows, i.e. resetting `naug` to
# `m` — no recomputation and no allocation.
function reset_projector!(P::SubspaceProjector)
    P.fixvars .= false
    empty!(P.fixidx)
    P.naug = P.m
    return
end

# Reset a coordinate subspace projector `P` by setting all components free.
# Elements of `fixvars` attribute are all set to false
function reset_projector!(P::CoordinateSubspaceProjector)

    P.fixvars .= false
    return
end

# Identify which bounds from the box `[max(-Δ,ℓ), min(Δ,u)]` become active at
# trial point `x + s` and set accordingly the coordinate subspace projector `P`.
# The components of the bounds identified as active are fixed for the rest of the current
# inner iteration
# Activity of bounds is measured up to small positive tolerance `eps_bound`.
function update_inner_active_set!(
    s::AbstractVector{T},
    slow::AbstractVector{T},
    supp::AbstractVector{T},
    P::Projector{T};
    eps_bound::T=sqrt(eps(T))) where T

    newly_active = Vector{Int}([])

    for i in axes(s, 1)
        if !is_fixed(P, i) &&
            (s[i] <= slow[i] + eps_bound * abs(slow[i]) || # at lower bound
            s[i] + eps_bound * abs(supp[i]) >= supp[i])    # at upper bound

            push!(newly_active, i)
        end
    end

    set_active!(P, newly_active)

    return
end

# Identify which bounds from the box `[ℓ, u]` are active at point `x` and set accordingly
# the subspace projector `P`.
# This is done to set up the projector for the computation of the criticality measure
# in the case where the linear constraints are general.
# The active set is rebuilt from scratch (reset + incremental appends), avoiding
# a full refactorization and any intermediate index array.
#
# On return
# `P` argument modified
function identify_active_set!(
    x::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T},
    P::SubspaceProjector{T};
    eps_bound::T = sqrt(eps(T))) where T

    reset_projector!(P)

    @inbounds for i in axes(x, 1)
        at_lower = isfinite(xlow[i]) && x[i] <= xlow[i] + abs(xlow[i]) * eps_bound
        at_upper = isfinite(xupp[i]) && x[i] + abs(xupp[i]) * eps_bound >= xupp[i]
        if at_lower || at_upper
            P.fixvars[i] = true
            append_active!(P, i)
        end
    end

    return
end

"""
    factor_to_boundary(p, s, sₗ, sᵤ, P)

Computes the largest scalar `γ` such that `s + γp` stays in the box `[sₗ,sᵤ]`.
The components considered are among free variables in a coordinate subspace
encoded in `Projector` `P`.
"""
function factor_to_boundary(
    p::Vector{T},
    s::Vector{T},
    s_l::Vector{T},
    s_u::Vector{T},
    proj_op::Projector{T}) where T

    stepmax = Inf
    eps_dir = T(1e-10)

    for i in axes(p, 1)
        if !is_fixed(proj_op, i)
            if p[i] < -eps_dir
                stepmax = min(stepmax, (s_l[i] - s[i]) / p[i])
            elseif p[i] > eps_dir
                stepmax = min(stepmax, (s_u[i] - s[i]) / p[i])
            end
        end
    end
    return stepmax
end


"""
    project!(v,x,ℓ,u)

Computes the projection of `x` onto the box `[ℓ,u]` and stores the results in `v`.
"""
function project!(v::Vector, x::Vector, x_low::Vector, x_upp::Vector)
    v[:] .= max.(x_low, min.(x, x_upp))
    return
end
