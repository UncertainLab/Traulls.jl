#=
    polyhedral_constraints.jl

Projection operators onto the null space of the linear equality constraints and active
bounds, and related methods for the handling of the polyhedral constraints.

Author(s): Pierre Borie
=#

"""
    SubspaceProjector{T}

This structure encodes the projector operator onto a subspace of the form
`{v | Av = 0, vᵢ = 0 for i ∈ 𝒜}`
where `A` is a full row rank `m × n` (`m < n`) matrix and the active set
`𝒜 = {i₁,...,iₚ}` (`p ≤ n - m`) is a subset of `{1,2,...,n}`.

The subspace is the null space of the matrix `A₊` defined as the concatenation of `A` with
the rows `e_iᵀ`, `i ∈ 𝒜`, of the `n × n` identity matrix. The projection of a vector `v`
is computed by the normal equations approach, i.e. `Pv = v - A₊ᵀy` with
`(A₊A₊ᵀ)y = A₊v`, using the block structure of the Cholesky factor of `A₊A₊ᵀ`:

```
L = [ L₁₁   0  ]
    [ Gᵀ   L_H ]
```

where `L₁₁` is the Cholesky factor of `AAᵀ`, `G = L₁₁⁻¹A_𝒜` gathers the columns of
`W = L₁₁⁻¹A` indexed by `𝒜` and `L_H` is the Cholesky factor of the Schur complement
`H = I - GᵀG`.

Since `A` never changes, `W` is formed once at construction and only the factor `L_H` is
maintained across the changes in the active set:

- activating a bound appends a column to `G`, i.e. a row and a column to `H`, and `L_H` is
  updated by a bordered update (see [`append_active!`](@ref));
- freeing a bound removes a row and a column from `H`, and `L_H` is updated by a rank-one
  update of its trailing block (see [`remove_active!`](@ref)).

The projection then reads `Pv = v - Wᵀ(z₁ - Gy₂) - E_𝒜y₂`, with `z₁ = Wv` and
`y₂ = H⁻¹(v_𝒜 - Gᵀz₁)`, so that no triangular solve with `L₁₁` is required.

A bound whose activation is implied by `Av = 0` and the bounds already active (redundant
constraint) is flagged as active but left out of the factor `L_H`, which keeps `H` positive
definite.

# Attributes

- `W`: `m × n` matrix `L₁₁⁻¹A`
- `fixvars`: `BitVector` of size `n` encoding the active bounds: `fixvars[i] = true` means
  that component `i` of vectors must equal `0` whereas `fixvars[i] = false` means that
  component `i` remains free
- `fixidx`: indices of the active bounds represented in the factor `L_H`, in the order of
  their activation. Row `k` of `L_H` corresponds to the bound on variable `fixidx[k]`
- `fixpos`: `Vector` of size `n` such that `fixpos[fixidx[k]] = k`, and `fixpos[i] = 0` if
  variable `i` is not represented in the factor `L_H`
- `LH`: buffer matrix whose leading `p × p` block stores the lower triangular factor `L_H`.
  It is enlarged when needed, up to the size `(n-m) × (n-m)`
- `p`: number of active bounds represented in the factor `L_H`
- `m`, `n`: dimensions of `A`
- `mbuf`: buffer vector of size `m`
- `pbuf`: buffer vector, of the same size as `LH`
- `pivot_tol`: tolerance on the squared pivot of the bordered update below which an
  activated bound is considered redundant
"""
mutable struct SubspaceProjector{T<:Real} <: Projector{T}
    W::Matrix{T}
    fixvars::BitVector
    fixidx::Vector{Int}
    fixpos::Vector{Int}
    LH::Matrix{T}
    p::Int
    m::Int
    n::Int
    mbuf::Vector{T}
    pbuf::Vector{T}
    pivot_tol::T
end

"""
    SubspaceProjector(A, chol_AAᵀ; pivot_tol = eps(T)^(2/3))

Constructor for the [`SubspaceProjector`](@ref) corresponding to the projection operator
onto the null space of the matrix `A`.

# Arguments

- `A`: full row rank `(m × n)` (`m < n`) matrix
- `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`

# Keywords

- `pivot_tol`: tolerance below which the squared pivot of a bordered update flags the
  activated bound as redundant. Defaults to `eps(T)^(2/3)`.
"""
function SubspaceProjector(
    A::AbstractMatrix{T},
    chol_aat::Cholesky{T};
    pivot_tol::T = eps(T)^(2/3)) where T

    (m,n) = size(A)
    m >= n && error("DimsError: The input matrix must have strictly less rows than columns")

    W = Matrix{T}(A)
    ldiv!(chol_aat.L, W) # W ← L₁₁⁻¹A

    cap = min(n-m, 16)

    SubspaceProjector{T}(W, falses(n), Int[], zeros(Int, n), zeros(T, cap, cap), 0, m, n,
                         Vector{T}(undef, m), Vector{T}(undef, cap), pivot_tol)
end

"""
    SubspaceProjector(A, fixvars, chol_AAᵀ; pivot_tol = eps(T)^(2/3))

Constructor for the [`SubspaceProjector`](@ref) corresponding to the projection operator
onto the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` (`m < n`) matrix.

# Arguments

- `A`: Linear equality matrix
- `fixvars`: `BitVector` encoding the vectors components that are set to 0
- `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`

# Keywords

- `pivot_tol`: tolerance below which the squared pivot of a bordered update flags the
  activated bound as redundant. Defaults to `eps(T)^(2/3)`.
"""
function SubspaceProjector(
    A::AbstractMatrix{T},
    fixvars::BitVector,
    chol_aat::Cholesky{T};
    pivot_tol::T = eps(T)^(2/3)) where T

    P = SubspaceProjector(A, chol_aat; pivot_tol = pivot_tol)
    set_active!(P, findall(fixvars))

    return P
end


"""
    CoordinateSubspaceProjector{T}

This structure encodes the projector operator onto a coordinate subspace of the form
`{v | vᵢ = 0 for i ∈ fixvars}`, where the components of vectors corresponding to active
bounds are set to `0`.

It is used instead of [`SubspaceProjector`](@ref) when the problem has no linear equality
constraints. The projection then only consists in zeroing the fixed components, so no
factorization is needed.

# Attributes

- `fixvars`: `BitVector` of size `n` encoding the fixed variables: `fixvars[i] = true`
  means that component `i` of vectors must equal `0` whereas `fixvars[i] = false` means
  that component `i` remains free
"""
mutable struct CoordinateSubspaceProjector{T<:Real} <: Projector{T}
    fixvars::BitVector
end

"""
    CoordinateSubspaceProjector(n; T = Float64)

Constructor for the [`CoordinateSubspaceProjector`](@ref) type.

Creates a projector with all `n` components free, i.e. with attribute `fixvars`
initialized to `falses(n)`.

# Arguments

- `n`: dimension of the space

# Keywords

- `T`: element type of the projector. Defaults to `Float64`.
"""
CoordinateSubspaceProjector(n::Int;T::DataType=Float64) = CoordinateSubspaceProjector{T}(falses(n))


"""
    mul!(r, P, v, α, β)

Computes `αPv + βr` and stores the result in `r`, where `P` is the projection operator
onto the subspace

`{v | Av = 0, vᵢ = 0 for i ∈ 𝒜}`

where `A` is a full row rank `m × n` (`m < n`) matrix and `𝒜 = {i₁,...,iₚ}`
(`p ≤ n - m`) is a subset of `{1,2,...,n}`.

The projection is computed as `Pv = v - Wᵀ(z₁ - Gy₂) - E_𝒜y₂`, where `z₁ = Wv` and `y₂`
solves `Hy₂ = v_𝒜 - Gᵀz₁` with the factor `L_H` (see [`SubspaceProjector`](@ref)).

Overloads the 5-argument `LinearAlgebra.mul!` method.

# Arguments

- `r`: Buffer vector to store the result. It may alias `v`
- `P`: Projection operator encoded as a [`SubspaceProjector`](@ref)
- `v`: input vector
- `α`, `β`: scalars

# On return

The vector `r`, containing `αPv + βr`.
"""
function mul!(
    r::AbstractVector{T},
    P::SubspaceProjector{T},
    v::AbstractVector{T},
    alpha::Number,
    beta::Number) where T

    W, z, p = P.W, P.mbuf, P.p
    y = view(P.pbuf, 1:p)

    mul!(z, W, v) # z ← z₁ = Wv

    if p > 0
        # y ← v_𝒜 - Gᵀz₁
        for k in 1:p
            i = P.fixidx[k]
            y[k] = v[i] - dot(view(W, :, i), z)
        end

        # y ← y₂ = H⁻¹(v_𝒜 - Gᵀz₁)
        L = LowerTriangular(view(P.LH, 1:p, 1:p))
        ldiv!(L, y)
        ldiv!(L', y)

        # z ← z₁ - Gy₂
        for k in 1:p
            axpy!(-y[k], view(W, :, P.fixidx[k]), z)
        end
    end

    # r ← α(v - Wᵀ(z₁ - Gy₂) - E_𝒜y₂) + βr
    if iszero(beta)
        r .= alpha .* v
    else
        r .= alpha .* v .+ beta .* r
    end
    mul!(r, W', z, -alpha, one(T))
    for k in 1:p
        r[P.fixidx[k]] -= alpha * y[k]
    end

    return r
end

"""
    mul!(r, P::SubspaceProjector, v)

Computes the projection `Pv` and stores the result in `r`, which may alias `v`.
Equivalent to `mul!(r, P, v, 1, 0)`.

# On return

The vector `r`, containing `Pv`.
"""
mul!(r::AbstractVector{T}, P::SubspaceProjector{T}, v::AbstractVector{T}) where T =
    mul!(r, P, v, one(T), zero(T))

"""
    Base.:*(P,x)

Computes the matrix-vector product `Px`, where `P` is the projection operator onto
the subspace `{v | Av = 0, vᵢ = 0 for i ∈ 𝒜}`
where `A` is a full row rank `m × n` (`m < n`) matrix
and `𝒜 = {i₁,...,iₚ}` (`p ≤ n - m`) is a subset of `{1,2,...,n}`.

Overloads the base multiplication `*` method.

# Arguments

- `P`: Projection operator encoded as a [`SubspaceProjector`](@ref)
- `x`: input vector

# On return

- `res`: `Vector` containing the result of the projection operation
"""
function Base.:*(P::SubspaceProjector{T}, x::AbstractVector{T}) where T

    res = Vector{T}(undef, size(x,1))
    mul!(res, P, x)
    return res
end

"""
    mul!(r, P::CoordinateSubspaceProjector, v, α, β)

Compute `αPv + βr`, where `Pv` is the projection of vector `v` onto the coordinate
subspace represented by `P`, and store the result in `r`.

The fixed components of `Pv` are `0` and the free ones are those of `v`, so that
`rᵢ ← βrᵢ` if `i` is fixed and `rᵢ ← αvᵢ + βrᵢ` otherwise.

Overloads the 5-argument `LinearAlgebra.mul!` method. 

# Arguments

- `r`: buffer vector to store the result. It may alias `v`
- `P`: projection operator encoded as a [`CoordinateSubspaceProjector`](@ref)
- `v`: input vector
- `α`, `β`: scalars

# On return

The vector `r`, containing `αPv + βr`.
"""
function mul!(
    r::AbstractVector{T},
    P::CoordinateSubspaceProjector{T},
    v::AbstractVector{T},
    alpha::Number,
    beta::Number) where T

    fixvars = P.fixvars
    if iszero(beta)
        for i in eachindex(r, v, fixvars)
            r[i] = fixvars[i] ? zero(T) : alpha * v[i]
        end
    else
        for i in eachindex(r, v, fixvars)
            r[i] = fixvars[i] ? beta * r[i] : alpha * v[i] + beta * r[i]
        end
    end

    return r
end

"""
    mul!(r, P::CoordinateSubspaceProjector, v)

Compute the projection `Pv` of vector `v` onto the coordinate subspace represented by `P`
and store the result in `r`, which may alias `v`. Equivalent to `mul!(r, P, v, 1, 0)`.

# On return

The vector `r`, containing `Pv`.
"""
mul!(r::AbstractVector{T}, P::CoordinateSubspaceProjector{T}, v::AbstractVector{T}) where T =
    mul!(r, P, v, one(T), zero(T))

"""
    Base.:*(P::CoordinateSubspaceProjector, v)

Compute the projection `Pv` of vector `v` onto the coordinate subspace represented by `P`
and return it in a newly allocated vector.

Overloads the base multiplication `*` method.

# Arguments

- `P`: projection operator encoded as a [`CoordinateSubspaceProjector`](@ref)
- `v`: input vector

# On return

- `res`: `Vector` containing the result of the projection
"""
function Base.:*(P::CoordinateSubspaceProjector, v::Vector)

    res = Vector{eltype(v)}(undef,size(v,1))
    mul!(res,P,v)

    return res
end


"""
    ensure_capacity!(P::SubspaceProjector, q)

Enlarge the buffers `LH` and `pbuf` of `P` so that they can hold a factor `L_H` of size
`q × q`. The capacity is at least doubled, up to `n - m`, and the current factor is
preserved.
"""
function ensure_capacity!(P::SubspaceProjector{T}, q::Int) where T

    cap = size(P.LH, 1)
    q <= cap && return

    newcap = min(max(2cap, q), P.n - P.m)
    LH = zeros(T, newcap, newcap)
    copyto!(view(LH, 1:P.p, 1:P.p), view(P.LH, 1:P.p, 1:P.p))
    P.LH = LH
    resize!(P.pbuf, newcap)

    return
end

"""
    append_active!(P::SubspaceProjector, j)

Add the bound on variable `j` to the factor `L_H` of `P` by a bordered update.

Adding `j` to the active set appends the column `wⱼ` (column `j` of `W`) to `G`, so that

```
H₊ = [ H   b ]      with  b = -Gᵀwⱼ  and  γ = 1 - ‖wⱼ‖².
     [ bᵀ  γ ]
```

The factor of `H₊` is obtained by appending the row `[ℓᵀ ρ]` to `L_H`, where `ℓ` solves
`L_H ℓ = b` and `ρ² = γ - ‖ℓ‖²`.

If the squared pivot `ρ²` is smaller than `P.pivot_tol`, the bound is implied by the 
constraints already in the subspace (redundant constraint) and the factor is left 
unchanged.

This function does not modify `P.fixvars`.

# On return

`true` if the bound has been added to the factor, `false` if it is redundant.
"""
function append_active!(P::SubspaceProjector{T}, j::Int) where T

    p = P.p

    # No degree of freedom left: any additional bound is redundant
    p == P.n - P.m && return false

    W = P.W
    wj = view(W, :, j)
    l = view(P.pbuf, 1:p)

    # l ← b = -Gᵀwⱼ
    for k in 1:p
        l[k] = -dot(view(W, :, P.fixidx[k]), wj)
    end
    gamma = one(T) - dot(wj, wj)

    # l ← L_H⁻¹b
    p > 0 && ldiv!(LowerTriangular(view(P.LH, 1:p, 1:p)), l)
    rho2 = gamma - dot(l, l)

    # Redundant constraint
    rho2 <= P.pivot_tol && return false

    ensure_capacity!(P, p+1)
    LH = P.LH
    for k in 1:p
        LH[p+1, k] = P.pbuf[k]
    end
    LH[p+1, p+1] = sqrt(rho2)

    push!(P.fixidx, j)
    P.fixpos[j] = p+1
    P.p = p+1

    return true
end

"""
    remove_active!(P::SubspaceProjector, k)

Remove from the factor `L_H` of `P` the bound represented by its `k`-th row.

Writing

```
L_H = [ L₁₁   0    0  ]
      [ l₂₁ᵀ  λ    0  ]  ← row k
      [ L₃₁   l₃₂  L₃₃ ]
```

the factor of `H` deprived of its `k`-th row and column is `[L₁₁ 0; L₃₁ L₃₃']`, where
`L₃₃'L₃₃'ᵀ = L₃₃L₃₃ᵀ + l₃₂l₃₂ᵀ` is computed by a rank-one update. 

This function does not modify `P.fixvars`.
"""
function remove_active!(P::SubspaceProjector{T}, k::Int) where T

    p, L, x = P.p, P.LH, P.pbuf
    q = p - k

    begin
        # x ← l₃₂
        for i in 1:q
            x[i] = L[k+i, k]
        end

        # Shift L₃₁ one row up and L₃₃ one row up and one column left
        for c in 1:k-1, i in k+1:p
            L[i-1, c] = L[i, c]
        end
        for c in k+1:p, i in c:p
            L[i-1, c-1] = L[i, c]
        end

        # Rank-one update of the trailing block: L₃₃'L₃₃'ᵀ = L₃₃L₃₃ᵀ + xxᵀ
        for i in 1:q
            d = L[k-1+i, k-1+i]
            rd = hypot(d, x[i])
            c, s = rd / d, x[i] / d
            L[k-1+i, k-1+i] = rd
            for l in i+1:q
                L[k-1+l, k-1+i] = (L[k-1+l, k-1+i] + s * x[l]) / c
                x[l] = c * x[l] - s * L[k-1+l, k-1+i]
            end
        end
    end

    P.fixpos[P.fixidx[k]] = 0
    deleteat!(P.fixidx, k)
    for t in k:p-1
        P.fixpos[P.fixidx[t]] = t
    end
    P.p = p-1

    return
end

"""
    append_redundant!(P::SubspaceProjector)

Try to add to the factor `L_H` the active bounds of `P` that were flagged as redundant.

After a bound is freed, a bound left out of the factor because it was implied by the other
constraints may no longer be redundant. It must then be represented in `L_H` for the
projection to keep the corresponding component equal to `0`.
"""
function append_redundant!(P::SubspaceProjector)

    for i in eachindex(P.fixvars)
        if P.fixvars[i] && P.fixpos[i] == 0
            append_active!(P, i)
        end
    end
    return
end


"""
    set_active!(proj_op, newly_active)

Add constraints `vᵢ = 0` for `i ∈ newly_active` to the subspace encoded in
`proj_op`, by applying a bordered update of the factor `L_H` for each bound that is not
already active (see [`append_active!`](@ref)).

# Arguments

- `proj_op`: [`SubspaceProjector`](@ref)
- `newly_active`: index, or collection of indices, of the variables that are set active
"""
function set_active!(P::SubspaceProjector, newly_active)

    for j in newly_active
        if !P.fixvars[j]
            P.fixvars[j] = true
            append_active!(P, j)
        end
    end
    return
end

"""
    set_active!(P::CoordinateSubspaceProjector, i::Int)

Fix the component at index `i` in the coordinate subspace represented by `P`, i.e. add
the constraint `vᵢ = 0`.
"""
@inline function set_active!(P::CoordinateSubspaceProjector, i::Int)
    P.fixvars[i] = true
end

"""
    set_active!(P::CoordinateSubspaceProjector, newly_fixed::Vector{Int})

Fix the components at indices in `newly_fixed` in the coordinate subspace represented by
`P`, i.e. add the constraints `vᵢ = 0` for `i ∈ newly_fixed`.
"""
@inline function set_active!(P::CoordinateSubspaceProjector, newly_fixed::Vector{Int})
    P.fixvars[newly_fixed] .= true
end


"""
    set_free!(proj_op, freevars)

Remove constraints `vᵢ = 0` for `i ∈ freevars` from the subspace encoded in
`proj_op`, by applying a rank-one update of the factor `L_H` for each bound that was
represented in it (see [`remove_active!`](@ref)).

The active bounds that were flagged as redundant are then added to the factor if they are
no longer implied by the other constraints (see [`append_redundant!`](@ref)).

# Arguments

- `proj_op`: [`SubspaceProjector`](@ref)
- `freevars`: index, or collection of indices, of the variables that are set free
"""
function set_free!(P::SubspaceProjector, freevars)

    removed = false
    for j in freevars
        if P.fixvars[j]
            P.fixvars[j] = false
            k = P.fixpos[j]
            if k > 0
                remove_active!(P, k)
                removed = true
            end
        end
    end

    removed && count(P.fixvars) > P.p && append_redundant!(P)
    return
end

"""
    set_free!(P::CoordinateSubspaceProjector, i::Int)

Free the component at index `i` in the coordinate subspace represented by `P`, i.e.
remove the constraint `vᵢ = 0`.
"""
@inline function set_free!(P::CoordinateSubspaceProjector, i::Int)
    P.fixvars[i] = false
end

"""
    set_free!(P::CoordinateSubspaceProjector, freed::Vector{Int})

Free the components at indices in `freed` in the coordinate subspace represented by `P`,
i.e. remove the constraints `vᵢ = 0` for `i ∈ freed`.
"""
@inline function set_free!(P::CoordinateSubspaceProjector, freed::Vector{Int})
    P.fixvars[freed] .= false
end


"""
    nb_degrees_of_freedom(proj_op::SubspaceProjector)

Return the number of degrees of freedom remaining in the subspace
`{v | Av = 0, vᵢ = 0 for i ∈ 𝒜}` represented by `proj_op`, i.e. `n - m - p` where
`A` is `m × n` and `p` is the number of active bounds represented in the factor `L_H`.
Redundant active bounds do not reduce the dimension of the subspace and are therefore not
counted.
"""
nb_degrees_of_freedom(P::SubspaceProjector) = P.n - P.m - P.p

"""
    nb_degrees_of_freedom(P::CoordinateSubspaceProjector)

Return the number of degrees of freedom remaining in the coordinate subspace represented
by `P`, i.e. the number of free variables.
"""
function nb_degrees_of_freedom(P::CoordinateSubspaceProjector)
    fixed = P.fixvars
    return size(fixed,1) - count(fixed)
end

"""
    saturated_subspace(P::Projector)

Return `true` if there are no degrees of freedom left in the subspace represented by the
projector operator `P`, `false` otherwise.
"""
saturated_subspace(P::Projector) = nb_degrees_of_freedom(P) == 0

"""
    is_fixed(proj_op::SubspaceProjector, i::Int)

Return `true` if the variable at index `i` is fixed in the subspace represented by
`proj_op`, `false` otherwise.
"""
is_fixed(P::SubspaceProjector, i::Int) = P.fixvars[i]

"""
    is_fixed(P::CoordinateSubspaceProjector, i::Int)

Return `true` if the variable at index `i` is fixed in the coordinate subspace represented
by `P`, `false` otherwise.
"""
is_fixed(P::CoordinateSubspaceProjector, i::Int) = P.fixvars[i]


"""
    reset_projector!(P::SubspaceProjector)

Reset the projector operator `P` by setting all bounds inactive.

All the variables are freed, so that `P` projects onto the null space of `A`. The factor
`L_H` is emptied without any computation.
"""
function reset_projector!(P::SubspaceProjector)

    P.fixvars .= false
    for k in 1:P.p
        P.fixpos[P.fixidx[k]] = 0
    end
    empty!(P.fixidx)
    P.p = 0
    return
end

"""
    reset_projector!(P::CoordinateSubspaceProjector)

Reset the coordinate subspace projector `P` by setting all the components free, i.e. all
the elements of attribute `fixvars` are set to `false`.
"""
function reset_projector!(P::CoordinateSubspaceProjector)

    P.fixvars .= false
    return
end

"""
    update_inner_active_set!(s, sₗ, sᵤ, P; eps_bound = sqrt(eps(T)))

Identify the bounds of the box `[sₗ, sᵤ]` that become active at the trial step `s` and fix
the corresponding variables in the projector operator `P`.

The step bounds are `sₗ = max(-Δ, xₗ - x)` and `sᵤ = min(Δ, xᵤ - x)`, so that a bound is
active either because `x + s` reaches a bound on the variables or because `s` reaches the
boundary of the `∞`-norm trust region. Only the variables that are not already fixed are
checked, and each newly active bound is fixed as soon as it is detected. The newly fixed 
variables stay fixed for the rest of the current inner iteration.

# Arguments

- `s`: trial step
- `sₗ`: lower bounds on the step `s`
- `sᵤ`: upper bounds on the step `s`
- `P`: `Projector` operator in which the newly active variables are fixed

# Keywords

- `eps_bound`: relative tolerance used to decide whether a bound is active. Defaults to
  the square root of the machine precision.

# On return

Nothing is returned, the projector `P` is modified in place.
"""
function update_inner_active_set!(
    s::AbstractVector{T},
    slow::AbstractVector{T},
    supp::AbstractVector{T},
    P::Projector{T};
    eps_bound::T=sqrt(eps(T))) where T

    for i in axes(s, 1)
        if !is_fixed(P, i) &&
            (s[i] <= slow[i] + eps_bound * abs(slow[i]) || # at lower bound
            s[i] + eps_bound * abs(supp[i]) >= supp[i])    # at upper bound

            set_active!(P, i)
        end
    end

    return
end

"""
    identify_active_set!(x, xₗ, xᵤ, P; eps_bound = sqrt(eps(T)))

Identify the bounds of the box `[xₗ, xᵤ]` that are active at point `x` and set the
subspace projector `P` accordingly.

This sets up the projector for the computation of the criticality measure when the
problem has linear equality constraints. Unlike [`update_inner_active_set!`](@ref), the
set of fixed variables is replaced, not extended: the projector is reset and the factor
`L_H` is rebuilt by successive bordered updates.

# Arguments

- `x`: current point
- `xₗ`: lower bounds on the variables `x`
- `xᵤ`: upper bounds on the variables `x`
- `P`: [`SubspaceProjector`](@ref) operator to set up

# Keywords

- `eps_bound`: relative tolerance used to decide whether a bound is active. Defaults to
  the square root of the machine precision.

# On return

Nothing is returned, the projector `P` is modified in place.
"""
function identify_active_set!(
    x::AbstractVector{T},
    xlow::AbstractVector{T},
    xupp::AbstractVector{T},
    P::SubspaceProjector{T};
    eps_bound::T = sqrt(eps(T))) where T

    reset_projector!(P)

    for i in axes(x, 1)
        at_lower = isfinite(xlow[i]) && x[i] <= xlow[i] + abs(xlow[i]) * eps_bound
        at_upper = isfinite(xupp[i]) && x[i] + abs(xupp[i]) * eps_bound >= xupp[i]
        (at_lower || at_upper) && set_active!(P, i)
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
    p::AbstractVector{T},
    s::AbstractVector{T},
    s_l::AbstractVector{T},
    s_u::AbstractVector{T},
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
    project!(x,ℓ,u)

Computes the projection of `x` onto the box `[ℓ,u]` and stores the results in `x`.
"""
function project!(
    x::AbstractVector{T}, 
    x_low::AbstractVector{T}, 
    x_upp::AbstractVector{T}) where T

    x[:] .= max.(x_low, min.(x, x_upp))
    return
end
