"""
    SubspaceMatrix{T}

This structure encodes the matrix that defines a subspace of the form
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`,
 (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

The subspace is merely the null space of the matrix `A₊` defined as the
concatenation of `A` with `Z` defined as the `p × n` matrix whose row `k` is the
row `iₖ` of the `n × n` identity matrix.

** Attributes

* `mat`: `AbstractMatrix` corresponding  to the linear equality constraints
matrix `A`

* `fixvars`: `BitVector` of size `n` encoding the matrix `Z`: `fixvars[i] = true`
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

** Argument

* `A::Matrix`: Full row rank matrix, `A` must have less rows than columns

** On return

* `SubspaceMatrix` with attribute `mat` set to `A` and `fixvars[i]` set to
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

** Attributes

* `mat`: `Transpose` corresponding  to the transpose of the linear equality
constraints matrix `A`

* `fixvars`: `BitVector` of size `n` encoding the fixed variables:
`fixvars[i] = true` means that components `i` of vectors must equal `0` whereas
`fixvars[i] = false` means that component `i` remains free
"""
struct TransposeSubspaceMatrix{T<:Real,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
    eqmat::Transpose{T,S}
    fixvars::BitVector
end

# Overloads the `transpose` function for `SubspaceMatrix`.
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

** Attributes

* `workspace_mat`: `SubspaceMatrix` representing matrix `A₊`

* `chol_gram_augmat`: `Factorization` storing the Cholesky decomposition of
`A₊A₊ᵀ`

* `chol_gram_eqmat`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
mutable struct SubspaceProjector{T<:Real} <: Projector{T}
    workspace_mat::SubspaceMatrix{T}
    chol_gram_augmat::Cholesky{T,Matrix{T}}
    chol_gram_eqmat::Cholesky{T,Matrix{T}}
end

"""
    SubspaceProjector(A,chol_AAᵀ)

Constructor for the `SubspaceProjector` corresponding to the projection operator
onto the null space of the matrix `A`.

** Arguments

* `A`: full row rank `(m × n)` (`m < n`) matrix

* `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
function SubspaceProjector(
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}}) where T

    SubspaceProjector(SubspaceMatrix(A),chol_aat,chol_aat)
end

"""
    SubspaceProjector

Constructor for the `SubspaceProjector` corresponding to the projection operator
onto the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and
`fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

** Arguments

* `A`: Linear equality matrix

* `fixvars`: `BitVector` encoding the vectors components that are set to 0

* `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
function SubspaceProjector(
    A::Matrix{T},
    fixvars::BitVector,
    chol_aat::Cholesky{T,Matrix{T}}) where T

    subA = SubspaceMatrix(A,fixvars)
    chol = cholesky_augmented_gram_mat(A,fixvars,chol_aat)

    SubspaceProjector(subA,chol,chol_aat)
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

# Overloads matrix vector product
Base.:*(M::SubspaceMatrix{T},x::Vector{T}) where T = vcat(M.eqmat*x,x[M.fixvars])

# overloads matrix vector product with transposition
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
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}` where `A` is a full row rank `m × n` ('m < n') matrix
and `fixvars = [i₁,...iₚ]` (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the `LinearAlgebra.mul!` method.

**Arguments**

* `r`: Buffer vector to store the result of the projection operation

* `P`: Projection operator encoded as a `SubspaceProjector`

* `x`: input vector

** On return

Nothing is returned, the result is stored in vector `r`.
"""
function mul!(r::Vector{T}, P::SubspaceProjector{T}, x::Vector{T}) where T

    temp = P.workspace_mat * x # form A₊x
    ldiv!(P.chol_gram_augmat, temp) # solve for y (A₊A₊ᵀ)y = A₊x
    r .= x .- transpose(P.workspace_mat)*temp # form r = x - A₊ᵀy

    return r
end

"""
    Base.:*(P,x)

Computes the matrix-vector product `Px`, where `P` is the projection operator onto
the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix
and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the base multiplication `*` method.

**Arguments**

* `P`: Projection operator encoded as a `SubspaceProjector`

* `x`: input vector

** On return

* `res`: `Vector` containing the result of the projection operation
"""
function Base.:*(P::SubspaceProjector{T}, x::Vector{T}) where T

    res = Vector{T}(undef, size(x,1))
    mul!(res, P, x)
    return res
end

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


# Update the Cholesky decomposition of the Gram matrix when adding one bound constraint 
# to the active set 
"""
    cholesky_augmented_gram_mat(A,fix_bounds,chol_AAᵀ)

Forms the Cholesky decomposition of the augmented Gram matrix `A₊A₊ᵀ`  with
`A₊` defined as the concatenation of `A`, full line rank `m × n`, with rows
of the `n × n` identity. The indices of the selected rows
`{i₁,...,iₚ} ⊂ {1,...n}`, with `p < n-m` are encoded into the `BitVector`
`fix_bounds`.

The computations exploits the block structure of `A₊A₊ᵀ` and the availability of
the Cholesky decomposition of `AAᵀ`.

**Arguments**

* `A`: full line rank matrix

* `fix_bounds`: `BitVector` encoding the fixed variables.
`fix_bounds[i] = true` means that a bound on component `i` is active

* `chol_AAᵀ`: Cholesky decomposition of `AAᵀ`

* On return

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
    L = LowerTriangular(Matrix{Float64}(undef, mpp, mpp))

    A_act_cols = view(A,:,fix_bounds)
    G = chol_aat.L \ A_act_cols
    mul!(H, G', G, -1, 1) # forms I - GᵀG

    # Forms the L factor of A₊A₊ᵀ Cholesy decomposition
    L[1:m,1:m] .= chol_aat.L
    L[m+1:end,1:m] .= G'
    L[m+1:end,m+1:end] .= cholesky(H).L

    return Cholesky(L)
end

"""
    update_subspace_projector!(proj_op, newly_active)

Add constraints `vᵢ = 0` for `i ∈ newly_active` to the subspace encoded in
`proj_op` and forms the corresponding projection operator by modifying the
Cholesky decomposition involved in the normal equations solving.

**Arguments**

* `proj_op`: `SubspaceProjector`

* `newly_active`: `Vector` containing the indices of the variables that are set
active
"""
function update_projector!(proj_op::SubspaceProjector, newly_active::Vector{Int})

    # Set new constraints active
    update_subspace!(proj_op.workspace_mat, newly_active)

    # Update the Cholesky decomposition involved in the normal equations solving
    proj_op.chol_gram_augmat = cholesky_augmented_gram_mat(
        proj_op.workspace_mat.eqmat,
        proj_op.workspace_mat.fixvars,
        proj_op.chol_gram_eqmat)
    return
end

# Set active the components of indices in `newly_active` into the projector
# `P`

@inline function update_projector!(P::CoordinateSubspaceProjector, newly_active::Vector{Int})

    P.fixvars[newly_active] .= true
    return
end

"""
    update_subspace!(M, newly_active)

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

# Returns the number of fixed variables in the subsspace represented by the `SubspaceMatrix` `A`
nb_fixed(submat::SubspaceMatrix) = count(submat.fixvars)


"""
    set_active!(proj_op, newly_active)

Add constraints `vᵢ = 0` for `i ∈ newly_active` to the subspace encoded in
`proj_op` and forms the corresponding projection operator by modifying the
Cholesky decomposition involved in the normal equations solving.

**Arguments**

* `proj_op`: `SubspaceProjector`

* `newly_active`: `Vector` containing the indices of the variables that are set
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

Remove constraints `vᵢ = 0` for `i ∈ freevars` to the subspace encoded in
`proj_op` and forms the corresponding projection operator by modifying the
Cholesky decomposition involved in the normal equations solving.

**Arguments**

* `proj_op`: `SubspaceProjector`

* `freevars`: `Vector` containing the indices of the variables that are set free
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

function nb_degrees_of_freedom(proj_op::SubspaceProjector)

    (m,n) = size(proj_op.workspace_mat.eqmat)

    return n - m - count(proj_op.workspace_mat.fixvars)
end

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
is_fixed(proj_op::SubspaceProjector, i::Int) = proj_op.workspace_mat.fixvars[i]

# Returns `true` if variable at index `i` is fixed, false if not
is_fixed(P::CoordinateSubspaceProjector, i::Int) = P.fixvars[i]


# Reset the projector operator by setting all bounds as inactive
function reset_projector!(P::SubspaceProjector)

    P.workspace_mat.fixvars .= false
    P.chol_gram_augmat = P.chol_gram_eqmat
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
# Activity of bounds is measured up to positive tolerance `atol`.

function update_active_set!(
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


# Identify which bounds from the box `[max(-Δ,ℓ), min(Δ,u)]` become active at
# trial point `x + s` and set accordingly the coordinate subspace projector `P`.
# Activity of bounds is measured up to positive tolerance `eps_active`.

# function update_active_set!(
#     s::Vector{T},
#     x::Vector{T},
#     xlow::Vector{T},
#     xupp::Vector{T},
#     P::CoordinateSubspaceProjector{T};
#     eps_bound::T=sqrt(eps(T))) where T

#     for i in axes(x,1)
#         P.fixvars[i] =  P.fixvars[i] ||
#             x[i] + s[i] <= xlow[i] + eps_bound*abs(xlow[i]) ||
#             x[i] + s[i] + eps_bound*abs(xupp[i]) >= xupp[i]
#     end
#     return
# end


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
