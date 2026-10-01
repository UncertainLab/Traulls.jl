#=
    polyhedral_constraints.jl

Projection operators onto the null space of the linear equality constraints and active
bounds, and related methods for the handling of the polyhedral constraints.

Author(s): Pierre Borie
=#

"""
    SubspaceMatrix{T}

This structure encodes the matrix that defines a subspace of the form
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`,
 (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

The subspace is merely the null space of the matrix `A₊` defined as the
concatenation of `A` with `Z` defined as the `p × n` matrix whose row `k` is the
row `iₖ` of the `n × n` identity matrix.

# Attributes

- `mat`: `AbstractMatrix` corresponding  to the linear equality constraints
  matrix `A`
- `fixvars`: `BitVector` of size `n` encoding the matrix `Z`: `fixvars[i] = true`
  means that components `i` of vectors must equal `0` whereas `fixvars[i] = false`
  means that component `i` remains free

`transpose` and base product `*` are overloaded for the type `SubspaceMatrix` in
order to make the computations with such a matrix efficient and without
explicitly storing the matrix `Z`.
"""
mutable struct SubspaceMatrix{T<:Real} <: AbstractMatrix{T} 
    eqmat::AbstractMatrix{T}
    fixvars::BitVector
end

"""
    SubspaceMatrix(A)

Constructor for the [`SubspaceMatrix`](@ref) type.
Creates a `SubspaceMatrix` where all variables are free.

# Argument

- `A::Matrix`: Full row rank matrix, `A` must have less rows than columns

# On return

- `SubspaceMatrix` with attribute `mat` set to `A` and `fixvars[i]` set to
  `false` for all `i`
"""
function SubspaceMatrix(A::Matrix{T}) where T
    (m,n) = size(A)
    if m >= n
        error("DimsError: The input matrix must have strictly less rows than columns")
    end

    SubspaceMatrix(A,falses(size(A,2)))
end

# Wrapper for the tranpose of a `SubspaceMatrix`
"""
    TransposeSubspaceMatrix{T,S}

Wrapper for the transpose of a [`SubspaceMatrix`](@ref).

# Attributes

- `mat`: `Transpose` corresponding  to the transpose of the linear equality
  constraints matrix `A`
- `fixvars`: `BitVector` of size `n` encoding the fixed variables:
  `fixvars[i] = true` means that components `i` of vectors must equal `0` whereas
  `fixvars[i] = false` means that component `i` remains free
"""
struct TransposeSubspaceMatrix{T<:Real,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
    eqmat::Transpose{T,S}
    fixvars::BitVector
end

"""
    transpose(M::SubspaceMatrix)

Return the transpose of `M` as a [`TransposeSubspaceMatrix`](@ref), without forming the
matrix `Z` explicitly.

Overloads the `LinearAlgebra.transpose` method.
"""
transpose(M::SubspaceMatrix{T}) where T =
    TransposeSubspaceMatrix(transpose(M.eqmat),
                            M.fixvars)


"""
    SubspaceProjector{T}

This structure encodes the projector operator onto a subspace of the form
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and
`fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

The subspace is the null space of the matrix `A₊` defined as the concatenation of
`A` with `Z`, a `p × n` matrix whose row `k` is the row `iₖ` of the `n × n`
identity matrix.

The projection is computed by solving the normal equations associated to the
projection quadratic program, which involves the Cholesky decomposition of
the augmented Gram matrix `A₊A₊ᵀ`.

# Attributes

- `workspace_mat`: `SubspaceMatrix` representing matrix `A₊`
- `chol_gram_augmat`: `Factorization` storing the Cholesky decomposition of
  `A₊A₊ᵀ`
- `chol_gram_eqmat`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
mutable struct SubspaceProjector{T<:Real} <: Projector{T}
    workspace_mat::SubspaceMatrix{T}
    chol_gram_augmat::Cholesky{T,Matrix{T}}
    chol_gram_eqmat::Cholesky{T,Matrix{T}}
end

"""
    SubspaceProjector(A,chol_AAᵀ)

Constructor for the [`SubspaceProjector`](@ref) corresponding to the projection operator
onto the null space of the matrix `A`.

# Arguments

- `A`: full row rank `(m × n)` (`m < n`) matrix
- `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
function SubspaceProjector(
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}}) where T

    SubspaceProjector(SubspaceMatrix(A),chol_aat,chol_aat)
end

"""
    SubspaceProjector(A,fixvars,chol_AAᵀ)

Constructor for the [`SubspaceProjector`](@ref) corresponding to the projection operator
onto the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and
`fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

# Arguments

- `A`: Linear equality matrix
- `fixvars`: `BitVector` encoding the vectors components that are set to 0
- `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
function SubspaceProjector(
    A::Matrix{T},
    fixvars::BitVector,
    chol_aat::Cholesky{T,Matrix{T}}) where T

    subA = SubspaceMatrix(A,fixvars)
    chol = cholesky_augmented_gram_mat(A,fixvars,chol_aat)

    SubspaceProjector(subA,chol,chol_aat)
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
    Base.:*(M::SubspaceMatrix, x)

Compute the matrix-vector product `A₊x`, where `A₊` is the matrix represented by `M`.

The result is the concatenation of `Ax` with the components `xᵢ` for `i ∈ fixvars`, which
avoids forming the matrix `Z` explicitly.

Overloads the base multiplication `*` method.
"""
Base.:*(M::SubspaceMatrix{T},x::Vector{T}) where T = vcat(M.eqmat*x,x[M.fixvars])

"""
    Base.:*(M::TransposeSubspaceMatrix, x)

Compute the matrix-vector product `A₊ᵀx`, where `A₊ᵀ` is the matrix represented by `M`.

The vector `x` has size `m + p`, where `p` is the number of fixed variables. The result is
`Aᵀx[1:m]`, to which the components of `x[m+1:end]` are added at the indices of the fixed
variables. This avoids forming the matrix `Zᵀ` explicitly.

Overloads the base multiplication `*` method.
"""
function Base.:*(A::TransposeSubspaceMatrix{T,S},x::Vector{T}) where {T,S}
    
    (n,m) = size(A.eqmat)
    res = Vector{T}(undef,n)
    
    mul!(res,A.eqmat,x[1:m])
    
    if any(A.fixvars)
        res[A.fixvars] .+= x[m+1:end]
    end

    return res
end


"""
    mul!(r, P, x)

Computes the matrix-vector product `Px` and stores the result in `r`, where `P`
is the projection operator onto the subspace

`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}` 
    
where `A` is a full row rank `m × n` ('m < n') 
matrix and `fixvars = [i₁,...iₚ]` (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the `LinearAlgebra.mul!` method.

# Arguments

- `r`: Buffer vector to store the result of the projection operation
- `P`: Projection operator encoded as a [`SubspaceProjector`](@ref)
- `x`: input vector

# On return

Nothing is returned, the result is stored in vector `r`.
"""
function mul!(r::Vector{T}, P::SubspaceProjector{T}, x::Vector{T}) where T

    temp = P.workspace_mat * x                  # form A₊x
    ldiv!(P.chol_gram_augmat, temp)             # solve for y (A₊A₊ᵀ)y = A₊x
    r .= x .- transpose(P.workspace_mat) * temp # form r = x - A₊ᵀy

    return r
end

"""
    Base.:*(P,x)

Computes the matrix-vector product `Px`, where `P` is the projection operator onto
the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix
and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the base multiplication `*` method.

# Arguments

- `P`: Projection operator encoded as a [`SubspaceProjector`](@ref)
- `x`: input vector

# On return

- `res`: `Vector` containing the result of the projection operation
"""
function Base.:*(P::SubspaceProjector{T}, x::Vector{T}) where T

    res = Vector{T}(undef, size(x,1))
    mul!(res, P, x)
    return res
end

"""
    mul!(r, P::CoordinateSubspaceProjector, v)

Compute the projection `Pv` of vector `v` onto the coordinate subspace represented by `P`
and store the result in `r`.

The fixed components of `r` are set to `0` and the free ones are copied from `v`.

Overloads the `LinearAlgebra.mul!` method.

# Arguments

- `r`: buffer vector to store the result of the projection
- `P`: projection operator encoded as a [`CoordinateSubspaceProjector`](@ref)
- `v`: input vector

# On return

Nothing is returned, the result is stored in vector `r`.
"""
function mul!(r::Vector, P::CoordinateSubspaceProjector, v::Vector)

    freevars = .!(P.fixvars)

    r[P.fixvars] .= 0          # set rᵢ = 0 for fixed components
    r[freevars] .= v[freevars] # set rᵢ = vᵢ for free components

    return
end

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
    cholesky_augmented_gram_mat(A,fix_bounds,chol_AAᵀ)

Forms the Cholesky decomposition of the augmented Gram matrix `A₊A₊ᵀ`  with
`A₊` defined as the concatenation of `A`, full line rank `m × n`, with rows
of the `n × n` identity. The indices of the selected rows
`{i₁,...,iₚ} ⊂ {1,...n}`, with `p < n-m` are encoded into the `BitVector`
`fix_bounds`.

The computations exploits the block structure of `A₊A₊ᵀ` and the availability of
the Cholesky decomposition of `AAᵀ`.

# Arguments

- `A`: full line rank matrix
- `fix_bounds`: `BitVector` encoding the fixed variables.
  `fix_bounds[i] = true` means that a bound on component `i` is active
- `chol_AAᵀ`: Cholesky decomposition of `AAᵀ`

# On return

The Cholesky decomposition `A₊A₊ᵀ` in a `Factorization` type.
"""
function cholesky_augmented_gram_mat(
    A::Matrix,
    fix_bounds::BitVector,
    chol_aat::Cholesky)

    (m,n) = size(A)
    p = count(fix_bounds)
    mpp = m+p

    # Auxiliary buffer arrays
    H = Matrix{Float64}(I,p,p)
    L = LowerTriangular(zeros(mpp, mpp))

    A_act_cols = view(A,:,fix_bounds)
    G = chol_aat.L \ A_act_cols
    mul!(H, G', G, -1, 1) # forms I - GᵀG

    # Forms the L factor of A₊A₊ᵀ Cholesy decomposition
    L[1:m,1:m] .= chol_aat.L
    L[m+1:end,1:m] .= G'
    L[m+1:end,m+1:end] .= cholesky(H).L

    return Cholesky(L)
end

# """
#     update_subspace_projector!(proj_op, newly_active)

# Add constraints `vᵢ = 0` for `i ∈ newly_active` to the subspace encoded in
# `proj_op` and forms the corresponding projection operator by modifying the
# Cholesky decomposition involved in the normal equations solving.

# **Arguments**

# * `proj_op`: `SubspaceProjector`

# * `newly_active`: `Vector` containing the indices of the variables that are set
# active
# """
# function update_projector!(proj_op::SubspaceProjector, newly_active::Vector{Int})

#     # Set new constraints active
#     update_subspace!(proj_op.workspace_mat, newly_active)

#     # Update the Cholesky decomposition involved in the normal equations solving
#     proj_op.chol_gram_augmat = cholesky_augmented_gram_mat(
#         proj_op.workspace_mat.eqmat,
#         proj_op.workspace_mat.fixvars,
#         proj_op.chol_gram_eqmat)
#     return
# end

# # Set active the components of indices in `newly_active` into the projector
# # `P`

# @inline function update_projector!(P::CoordinateSubspaceProjector, newly_active::Vector{Int})

#     P.fixvars[newly_active] .= true
#     return
# end

"""
    set_subspace!(M, active)

Set the constraints `vᵢ = 0`, for `i ∈ active` in the subspace represented by matrix
`M`.
"""
function set_subspace!(M::SubspaceMatrix, active::Vector{Int})
    M.fixvars .= false
    M.fixvars[active] .= true
    return
end
"""
    add_subspace!(M, newly_active)

Add the constraints `vᵢ = 0`, for `i ∈ newly_active` to the subspace represented by matrix
`M`. Corresponds to adding rows to the latter.
"""
function add_subspace!(M::SubspaceMatrix, newly_active::Vector{Int})

    M.fixvars[newly_active] .= true
    return
end

"""
    remove_subspace!(M, removed)

Remove the constraints `vᵢ = 0`, for `i ∈ removed` from the subspace represented by matrix
`M`. Corresponds to removing rows from the latter.
"""
function remove_subspace!(M::SubspaceMatrix, removed::Vector{Int})
    M.fixvars[removed] .= false
    return
end

"""
    nb_fixed(M::SubspaceMatrix)

Return the number of fixed variables in the subspace represented by `M`.
"""
nb_fixed(submat::SubspaceMatrix) = count(submat.fixvars)


"""
    set_active!(proj_op, newly_active)

Add constraints `vᵢ = 0` for `i ∈ newly_active` to the subspace encoded in
`proj_op` and forms the corresponding projection operator by modifying the
Cholesky decomposition involved in the normal equations solving.

# Arguments

- `proj_op`: [`SubspaceProjector`](@ref)
- `newly_active`: `Vector` containing the indices of the variables that are set
  active
"""
function set_active!(proj_op::SubspaceProjector, newly_active::Vector{Int})

    # Set new constraints active
    add_subspace!(proj_op.workspace_mat, newly_active)

    # Update the Cholesky decomposition involved in the normal equations solving
    proj_op.chol_gram_augmat = cholesky_augmented_gram_mat(
        proj_op.workspace_mat.eqmat,
        proj_op.workspace_mat.fixvars,
        proj_op.chol_gram_eqmat)
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

Remove constraints `vᵢ = 0` for `i ∈ freevars` to the subspace encoded in
`proj_op` and forms the corresponding projection operator by modifying the
Cholesky decomposition involved in the normal equations solving.

# Arguments

- `proj_op`: [`SubspaceProjector`](@ref)
- `freevars`: `Vector` containing the indices of the variables that are set free
"""
function set_free!(proj_op::SubspaceProjector, freevars::Vector{Int})
    # Set new constraints active
    remove_subspace!(proj_op.workspace_mat, freevars)

    # Update the Cholesky decomposition involved in the normal equations solving
    proj_op.chol_gram_augmat = cholesky_augmented_gram_mat(
        proj_op.workspace_mat.eqmat,
        proj_op.workspace_mat.fixvars,
        proj_op.chol_gram_eqmat)
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
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}` represented by `proj_op`, i.e. `n - m - p` where
`A` is `m × n` and `p` is the number of fixed variables.
"""
function nb_degrees_of_freedom(proj_op::SubspaceProjector)

    (m,n) = size(proj_op.workspace_mat.eqmat)

    return n - m - count(proj_op.workspace_mat.fixvars)
end

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
is_fixed(proj_op::SubspaceProjector, i::Int) = proj_op.workspace_mat.fixvars[i]

"""
    is_fixed(P::CoordinateSubspaceProjector, i::Int)

Return `true` if the variable at index `i` is fixed in the coordinate subspace represented
by `P`, `false` otherwise.
"""
is_fixed(P::CoordinateSubspaceProjector, i::Int) = P.fixvars[i]


"""
    reset_projector!(P::SubspaceProjector)

Reset the projector operator `P` by setting all bounds inactive.

All the variables are freed, so that `P` projects onto the null space of `A`. The Cholesky
decomposition of the augmented Gram matrix is reset to the one of `AAᵀ`.
"""
function reset_projector!(P::SubspaceProjector)

    P.workspace_mat.fixvars .= false
    P.chol_gram_augmat = P.chol_gram_eqmat
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
checked. The newly fixed variables stay fixed for the rest of the current inner
iteration.

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

"""
    identify_active_set!(x, xₗ, xᵤ, P; eps_bound = sqrt(eps(T)))

Identify the bounds of the box `[xₗ, xᵤ]` that are active at point `x` and set the
subspace projector `P` accordingly.

This sets up the projector for the computation of the criticality measure when the
problem has linear equality constraints. Unlike [`update_inner_active_set!`](@ref), the
set of fixed variables is replaced, not extended, and the Cholesky decomposition of the
augmented Gram matrix `A₊A₊ᵀ` is recomputed.

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

    # TODO: allocate `active` array in `solve!` method and pass as argument modified in
    # place within this function
    active = Vector{Int}([])

    # Identify active bounds
    for i in axes(x, 1)
        at_lower = isfinite(xlow[i]) && x[i] <= xlow[i] + abs(xlow[i]) * eps_bound
        at_upper = isfinite(xupp[i]) && x[i] + abs(xupp[i]) * eps_bound >= xupp[i]
        (at_lower || at_upper) && push!(active, i)
    end

    # Set constraints active
    set_subspace!(P.workspace_mat, active)

    # Update the Cholesky decomposition involved in the normal equations solving
    P.chol_gram_augmat = isempty(active) ? P.chol_gram_eqmat :
        cholesky_augmented_gram_mat(
        P.workspace_mat.eqmat,
        P.workspace_mat.fixvars,
        P.chol_gram_eqmat)

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
