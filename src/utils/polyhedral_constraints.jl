"""
    SubspaceMatrix{T}

This structure encodes the matrix that defines a subspace of the form `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

The subspace is merely the null space of the matrix `A₊` defined as the concatenation of `A` with `Z` defined as the
`p × n` matrix whose row `k` is the row `iₖ` of the `n × n` identity matrix.

** Attributes

* `mat`: `AbstractMatrix` corresponding  to the linear equality constraints matrix `A`

* `fixvars`: `BitVector` of size `n` encoding the matrix `Z`: `fixvars[i] = true` means that components `i`
of vectors must equal `0` whereas `fixvars[i] = false` means that component `i` remains free

`transpose` and base product `*` are overloaded for the type `SubspaceMatrix` in order to make the computations
with such a matrix efficient and without explicitly storing the matrix `Z`.
"""
mutable struct SubspaceMatrix{T<:Real} <: AbstractMatrix{T} 
    mat::AbstractMatrix{T}
    fixvars::BitVector
end

"""
    SubspaceMatrix(A)

Constructor for the [`SubspaceMatrix`](@ref) type.
Creates a `SubspaceMatrix` where all variables are free.

** Argument

* `A::Matrix`: Full row rank matrix, `A` must have less rows than columns

** On return

* `SubspaceMatrix` with attribute `mat` set to `A` and `fixvars[i]` set to `false` for all `i`
"""
function SubspaceMatrix(A::Matrix{T}) where T
    (m,n) = size(A)
    @assert m < n "The input matrix must have strictly less rows than columns"

    SubspaceMatrix(A,falses(size(A,2)))
end

# Wrapper for the tranpose of a `SubspaceMatrix`
"""
    TransposeSubspaceMatrix{T,S}

Wrapper for the transpose of a [`SubspaceMatrix`](@ref).

** Attributes

* `mat`: `Transpose` corresponding  to the transpose of the linear equality constraints matrix `A`

* `fixvars`: `BitVector` of size `n` encoding the fixed variables: `fixvars[i] = true` means that components `i`
of vectors must equal `0` whereas `fixvars[i] = false` means that component `i` remains free

"""
struct TransposeSubspaceMatrix{T<:Real,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
    mat::Transpose{T,S}
    fixvars::BitVector
end

# Overloads the `transpose` function for `SubspaceMatrix`.
transpose(submat::SubspaceMatrix{T}) where T = TransposeSubspaceMatrix(transpose(submat.mat), submat.fixvars)

# Overloads matrix vector product
Base.:*(A::SubspaceMatrix{T},x::Vector{T}) where T = vcat(A.mat*x,x[A.fixvars])

# overloads matrix vector product with transposition
function Base.:*(A::TransposeSubspaceMatrix{T,S},x::Vector{T}) where {T,S}
    
    (n,m) = size(A.mat)
    res = Vector{T}(undef,n)
    
    mul!(res,A.mat,x[1:m])
    
    if any(A.fixvars)
        res[A.fixvars] .+= x[m+1:end]
    end

    return res
end

# Returns the number of fixed variables in the subsspace represented by the `SubspaceMatrix` `A`
nb_fixed(submat::SubspaceMatrix{T}) where T = count(submat.fixvars)

#= Mutable struct to encode the operator that computes projections onto the subspace
`{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`,
using the normal equations approach.
=#

"""
    SubspaceProjector{T}

This structure encodes the projector operator onto a subspace of the form `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

The subspace is merely the null space of the matrix `A₊` defined as the concatenation of `A` with `Z` defined as the
`p × n` matrix whose row `k` is the row `iₖ` of the `n × n` identity matrix.

The projection is computed by solving the normal equations associated to the projection quadratic program,
which involves the Cholesky decomposition of the augmented matrix `A₊A₊ᵀ`.

** Attributes

* `mat`: `SubspaceMatrix` representing matrix `A₊`

* `chol`: `Factorization` storing the Cholesky decomposition of `A₊A₊ᵀ`
"""
mutable struct SubspaceProjector{T<:Real} <: Projector{T}
    mat::SubspaceMatrix{T}
    chol::Cholesky{T,Matrix{T}}
end

# Constructor for polyhedra with no active bounds
"""
    SubspaceProjector(A,chol_AAᵀ)

Constructor for the `SubspaceProjector` corresponding to the projection operator onto the null space of the matrix `A`.

** Arguments

* `A`: full row rank `(m × n)` (`m < n`) matrix

* `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
SubspaceProjector(A::Matrix{T},chol::Cholesky{T,Matrix{T}}) where T = SubspaceProjector(SubspaceMatrix(A),chol)

# Constructor for polyhedra with some active bounds
"""
    SubspaceProjector

Constructor for the `SubspaceProjector` corresponding to the projection operator onto the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

** Arguments

* `A`: Linear equality matrix

* `fixvars`: `BitVector` encoding the vectors components that are set to 0

* `chol_AAᵀ`: `Factorization` storing the Cholesky decomposition of `AAᵀ`
"""
function SubspaceProjector(A::Matrix{T},fixvars::BitVector,chol_aat::Cholesky{T,Matrix{T}}) where T

    subA = SubspaceMatrix(A,fixvars)
    chol = cholesky_aug_aat(A,fixvars,chol_aat)
    
    SubspaceProjector(subA,chol)
end

# Overload of the `mul!` method to make projections behave as matrix-vector products
"""
    mul!(r,P,x)

Computes the matrix-vector product `Px` and stores the result in `r`, where `P` is the projection operator onto
the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the `LinearAlgebra.mul!` method.

** Arguments

* `r`: Buffer vector to store the result of the projection operation

* `P`: Projection operator encoded as a `SubspaceProjector`

* `x`: input vector

** On return

Nothing is returned, the result is stored in vector `r`.
"""
function mul!(r::Vector{T},P::SubspaceProjector{T},x::Vector{T}) where T

    temp = P.mat*x # form Ãv
    ldiv!(P.chol,temp) # solve for y (ÃÃᵀ)y = Ãv
    r .= x .- transpose(P.mat)*temp # form r = v - Ãᵀy
    
    return r 
end

"""
    Base.:*(P,x)

Computes the matrix-vector product `Px`, where `P` is the projection operator onto
the subspace `{v | Av = 0, vᵢ = 0 for i ∈ fixvars}`
where `A` is a full row rank `m × n` ('m < n') matrix and `fixvars = [i₁,...iₚ]`, (`p ≤ n - m`) is a subset of `[1,2,...,n]`.

Overloads the base multiplication `*` method.

** Arguments

* `P`: Projection operator encoded as a `SubspaceProjector`

* `x`: input vector

** On return

* `res`: `Vector` containing the result of the projection operation
"""
function Base.:*(P::SubspaceProjector{T}, x::Vector{T}) where T 

    res = Vector{T}(undef,size(x,1))
    mul!(res,P,x)
    return res 
end

# Reset the projector operator by setting all bounds as inactive 
function reset_projector!(P::SubspaceProjector{T}, chol_aat::Cholesky{T,Matrix{T}}) where T 

    P.mat.fixvars .= false
    P.chol = chol_aat 
    return
end
# Add the constraints `xᵢ=0 for i ∈ idx` to the subspace matrix 
# Modifes accordingly the projection operator    
# Temporary version using the Cholesky decomposition of the initial augmented matrix 
function add_active_bounds!(P::SubspaceProjector, idx::Vector{Int},chol_aat::Cholesky)
    P.mat.fixvars[idx] .= true
    P.chol = cholesky_aug_aat(P.mat.mat,P.fixvars,chol_aat)
    return
end

# Add the constraint `xᵢ=0 for some i` to the subspace matrix 
# Modifes accordingly the projection operator    
# Temporary version using the Cholesky decomposition of the initial augmented matrix 
function add_active_bound!(P::SubspaceProjector, idx::Int, chol_aat::Cholesky)
    add_active_bounds!(P, [idx], chol_aat)
    return
end

# Returns the number of fixed variables in the subspace the operator `P` projects on
nb_fixed(P::SubspaceProjector) = nb_fixed(P.mat)

# Identify the bounds active at `x` up to `atol` and update the projection operator

function active_bounds!(
    x::Vector{T},
    P::SubspaceProjector{T},
    chol_aat::Cholesky,
    x_low::Vector{T},
    x_upp::Vector{T};
    atol::Float64 = sqrt(eps(T))) where T



    active = BitVector(undef,size(x,1))
    for i in axes(x,1)
        active[i] = P.mat.fixvars[i] || (x[i] <= x_low[i] + atol) || (x_upp[i] - atol <= x[i])
    end
    add_active_bounds(P,findall(active),chol_aat)
    return
end

""" MixedConstraints <: PolyhedralConstraints

This mutablle structure represents and encodes a polyhedral set with linear equalities, bounds and some variables fixed to `0` of the form

`{v | Av = 0, xₗ ≤ v ≤ xᵤ, ∀i ∈ 𝒜, vᵢ = 0}`

**Attributes**

* `lineq` matrix `A` from the linear equality constraints

* `xlow` and `xupp` respectively represent lower and upper bounds `xₗ` and `xᵤ`

* `fixvars` is a `BitVector` encoding the set of fixed variables `𝒜`, i.e. `fixvars[i]=true → vᵢ = 0`
"""
mutable struct MixedConstraints <: PolyhedralConstraints
    lineq::Matrix
    xlow::Vector
    xupp::Vector
    fixvars::BitVector
    chol::Cholesky
end

function MixedConstraints(
        A::Matrix,
        chol_aat::Cholesky; 
        l::Vector = fill(-Inf,size(A,2)), 
        u::Vector=fill(Inf,size(A,2)))
    
    fixed = BitVector(undef,size(A,2))
    fixed .= false
    MixedConstraints(A,l,u,fixed,chol_aat)
end   

function MixedConstraints(
        A::Matrix,
        chol_aat::Cholesky,
        fixed::BitVector; 
        l::Vector = fill(-Inf,size(A,2)), 
        u::Vector=fill(Inf,size(A,2))) 
    
    chol = (any(fixed) ? cholesky_aug_aat(A, fixed, chol_aat) : chol_aat)
    MixedConstraints(A,l,u,fixed,chol)
end  

nb_fix(lincons::MixedConstraints) = count(lincons.fixvars)
#= Forms the Cholesky decomposition of ÃÃᵀ 
Computation exploits its block structure =#

function cholesky_aug_aat(
    A::Matrix, 
    fix_bounds::BitVector, 
    chol_aat::Cholesky) 

    (m,n) = size(A)
    p = count(fix_bounds)
    mpp = m+p
    @assert mpp <= n 

    # Auxiliary buffer arrays 
    H = Matrix{Float64}(I,p,p)
    L = LowerTriangular(Matrix{Float64}(undef, mpp, mpp))
    
    A_act_cols = view(A,:,fix_bounds)
    G = chol_aat.L \ A_act_cols
    # TODO: implement a more efficient computation of I - GᵀG
    mul!(H, G', G, -1, 1) # forms I - GᵀG
    
    # Forms the L factor of ÃÃᵀ Cholesy decomposition
    L[1:m,1:m] .= chol_aat.L
    L[m+1:end,1:m] .= G'
    L[m+1:end,m+1:end] .= cholesky(H).L  
    return Cholesky(L)
end

# Perform the Cholesky decomposition update on the representation `lincons`
function update_chol!(
        lincons::MixedConstraints, 
        chol_aat::Cholesky) 
    
    lincons.chol = cholesky_aug_aat(lincons.lineq, lincons.fixvars, chol_aat)
    return
end

#= The two following methods perform respectively the left multiplication by `Ã` and `Ãᵀ` =#

function left_mul_tr(lincons::MixedConstraints, y::Vector) 
    
    (m,n) = size(lincons.lineq)
    x = Vector(undef,n)
    
    if any(lincons.fixvars)
        mul!(x,lincons.lineq',y[1:m])
        x[lincons.fixvars] .+= y[m+1:end]
    else
        mul!(x,lincons.lineq',y)
    end
    return x
end

function left_mul(lincons::MixedConstraints, x::Vector) 
    
    (m,_) = size(lincons.lineq)
    y = Vector(undef, m+count(lincons.fixvars))

    if any(lincons.fixvars)
        mul!(view(y,1:m), lincons.lineq, x)
        y[m+1:end] .= x[lincons.fixvars]
    else
        mul!(y, lincons.lineq, x)
    end
    return y
end
    

#=
In place versions of the projection methods onto, respectively, the nullspace of `A` and sets of the form `{v | Av = 0, vᵢ = 0 for i ∈ fixed variables}`
=#
function projection_nullspace!(
    lincons::MixedConstraints,
    r::Vector,
    v::Vector
    ) 

    @assert !any(lincons.fixvars)
    m = size(lincons.lineq,1)
    w, y = Vector(undef,m), Vector(undef,m) # auxiliary vectors

    y[:] = lincons.chol.L \ (lincons.lineq*r)
    w[:] = lincons.chol.U \ y
    v[:] = r - lincons.lineq'*w
    return
end

function projection_subspace!(
    lincons::MixedConstraints, 
    r::Vector,
    v::Vector
    ) 

    (m,n) = size(lincons.lineq)
    mpp = m + count(lincons.fixvars)
    @assert m < mpp <= n 

    w, y = Vector(undef,mpp), Vector(undef,mpp) # auxiliary vectors
    
    y[:] = lincons.chol.L \ left_mul(lincons,r) 
    w[:] = lincons.chol.U \ y 
    v[:] = r - left_mul_tr(lincons,w)
    return
end


"""
    projection(lincons,r)

Computes and returns the projection of vector `r` onto the null space of a matrix `Ã` represented by `lincons` (see [`MixedConstraints`](@ref)).

The nullspace of `Ã` corresponds to the feasible set `{v | Av = 0, vᵢ = 0 for i ∈ lincons.fixvars}`.

The projection is computed by solving the normal equations associated to the problem `minᵥ {||v-r|| | Ãv = 0}` using the Cholesky factorization of `ÃÃᵀ`.

If there are no fixed variables, i.e. `Ã = A`, then simply perfoms the projection onto the nullspace of `A`.
"""
function projection(lincons::MixedConstraints, r::Vector) 
        
    v = Vector{Float64}(undef,size(r,1))
    projection!(lincons,r,v)
    return v
end

# In place version of the above `projection` method
function projection!(
    lincons::MixedConstraints, 
    r::Vector, 
    v::Vector
    ) 

    if any(lincons.fixvars)
        projection_subspace!(lincons,r,v)
    else
        projection_nullspace!(lincons,r,v)
    end
    return
end

"""
    function projection_polyhedron(x,A,b,l,u;solver)

Compute the projection of vector 'x' onto the polyhedron '{v | Av = b, l ≤ v ≤ u}' by solving the associated minimum-distance quadratic program.

The default solver is 'Ipopt'.
"""
function projection_polyhedron(
    x::Vector, 
    A::Matrix, 
    b::Vector, 
    l::Vector, 
    u::Vector; 
    solver = Ipopt.Optimizer) 

    n = size(x,1)
    model = Model(solver)
    set_silent(model)
    set_attribute(model, "hessian_constant", "yes") # Option to make Ipopt assume it is a QP

    @variable(model, l[i] <= v[i=1:n] <= u[i])
    @constraint(model, A*v .== b)
    @objective(model, Min, dot(v-x,v-x))
    optimize!(model)

    return value.(v)
end


#= Identify the bounds active at `x` up to `atol` and update the Cholesky decomposition used to compute projections =#

function active_bounds!(
        lincons::MixedConstraints,
        x::Vector,
        chol_aat::Cholesky;
        atol::Float64 = sqrt(eps(Float64))
        ) 

    for i in axes(x,1)
        lincons.fixvars[i] = (x[i]-lincons.xlow[i] <= atol) || (lincons.xupp[i]-x[i] <= atol)
    end
    update_chol!(lincons,chol_aat)
    return
end


#= Identify the bounds that become active when taking the step `s` from `x` in the intersection of the feasible domain and the trust region (up to `atol`) 
Update the Cholesky decomposition used to compute projections on the resulting subspace =#
function active_bounds(
    lincons::MixedConstraints,
    x::Vector,
    s::Vector,
    delta::Float64;
    atol::Float64 = sqrt(eps(Float64))
    ) 

    s_l = (t -> max(t,-delta)).(lincons.xlow-x)
    s_u = (t -> min(t,delta)).(lincons.xupp-x)
    at_bound = BitVector(undef,size(x,1))

    for i in axes(s,1)
        at_bound[i] = (s[i]-s_l[i] <= atol) || (s_u[i]-s[i] <= atol)
    end

    active = findall(at_bound)
    return active
end
#= Set the variable `ind` to active and update the Cholesky factorization =#

function add_active!(
        lincons::MixedConstraints,
        chol_aat::Cholesky,
        ind::Int
        ) 

    lincons.fixvars[ind] = true
    update_chol!(lincons,chol_aat)
    return
end

#= Set the variables in `indx` to active and update the Cholesky factorization =#
function add_active!(
        lincons::MixedConstraints,
        chol_aat::Cholesky,
        indx::Vector{Int}
        ) 

    lincons.fixvars[indx] .= true
    update_chol!(lincons,chol_aat)
    return
end

""" BoundConstraints

Mutable struct representing a set of lower and upper bounds and which ones are considered active.

**Attributes**

* `xlow` and `xupp` are double precision vectors that respectively represent lower and upper bounds

* `fixvars` is a `BitVector` encoding the indices of active bounds `𝒜`, i.e. `fixvars[i]=true → vᵢ = 0`
"""
mutable struct BoundConstraints <: PolyhedralConstraints
    xlow::Vector
    xupp::Vector
    fixvars::BitVector
end

function BoundConstraints(xlow::Vector, xupp::Vector) 
    fixed = BitVector(undef,size(xlow,1))
    fixed .= false
    BoundConstraints(xlow,xupp,fixed)
end

"""
    project!(v,x,ℓ,u)

Computes the projection of `x` onto the box `[ℓ,u]` and stores the results in `v`.
"""
function project!(v::Vector, x::Vector, x_low::Vector, x_upp::Vector) 
    v[:] .= max.(x_low, min.(x,x_upp))
    return
end


"""
    sort_breakpoints(x,g,ℓ,u,Δ;atol) -> unique_vals, grouped_indices

Computes the breakpoints of the projected gradient path `-tg` for `t ≥ 0` onto the box `{d | min(ℓ-x,-Δe) ≤ d ≤ max(u-x,Δe)}`.
Breakpoints are then sorted in ascending order with duplicates removed. 
For each unique breakpoint, the function returns the list of indices (from the 
original array) his value occurs.

# Returns
- `unique_vals`: Sorted vector of unique breakpoints.
- `grouped_indices`: A vector where entry `i` contains the indices of the variables associated to breakpoint number `i`.
  the indices of `values` corresponding to the matching entry in `unique_vals`.
"""
function sort_breakpoints(
    x::Vector,
    g::Vector,
    x_low::Vector,
    x_upp::Vector,
    delta::Float64;
    atol = sqrt(eps(Float64)))  

    
    n = size(x,1)
    breakpoints = Vector{Float64}(undef,n)

    # Compute the breakpoints
    for i=1:n
        if g[i] > atol
            breakpoints[i] = max(x_low[i]-x[i],-delta) / -g[i]
        elseif g[i] < -atol
            breakpoints[i] = min(x_upp[i]-x[i],delta) / -g[i]
        else
            breakpoints[i] = 0.0
        end 
    end

    

    # Form sorted breakpoints values and corresponding indices 
    sorted_breakpoints, grouped_indices = group_breakpoints(breakpoints)

    return sorted_breakpoints, grouped_indices
end

# Form sorted breakpoints values and corresponding indices 
function group_breakpoints(breakpoints::Vector) 
    
    idx = sortperm(breakpoints)              # indices that sort the values
    sorted_vals = breakpoints[idx]           # sorted values
    
    # Collect unique values and group corresponding indices
    groups = Dict{Float64, Vector{Int}}()
    for (v, i) in zip(sorted_vals, idx)
        push!(get!(groups, v, Int[]), i)
    end
    
    unique_vals = collect(keys(groups)) |> sort
    grouped_indices = [groups[v] for v in unique_vals]
    
    return unique_vals, grouped_indices
end

"""
    projection(x,ℓ,u)

Computes and returns the projection of `x` onto the box constrainted set `{v | ℓ ≤ v ≤ u}`.
"""
function project(x::Vector, x_low::Vector, x_upp::Vector)  
    
    v = Vector(undef,size(x,1))
    project!(v,x,x_low, x_upp)
    return v 
end

#= Identify the bounds that become active when taking the step `s` from `x` in the intersection of the feasible domain and the trust region (up to `atol`) 
Update the Cholesky decomposition used to compute projections on the resulting subspace =#

"""
    active_idx(x,ℓ,u;atol)

Identifies the bounds active at `x` using tolerance `atol`. Returns the corresponding list of indices encoded in a BitVector
"""
function active_idx(
    x::Vector,
    x_low::Vector,
    x_upp::Vector; 
    atol::Float64=sqrt(eps(Float64)))  

    at_bound = BitVector(undef,size(x,1))

    for i in axes(x,1)
        at_bound[i] = (x[i]-x_low[i] <= atol) || (x_upp[i]-x[i] <= atol)
    end

    active = findall(at_bound)
    return active
end

function initial_active_bounds(
    x::Vector,
    d::Vector,
    x_low::Vector,
    x_upp::Vector;
    atol = sqrt(eps(Float64)))

    fix_vars = falses(size(x,1))

    for i in axes(x,1)
        
        fix_vars[i] = (x_upp[i]-atol < x[i] && d[i] > atol) || # positive direction at active upper bound 
            (x_low[i]+atol > x[i] && d[i] < -atol) ||          # negative direction at active lower bound 
            (abs(d[i]) < atol)                                 # zero direction
    end

    return findall(fix_vars)
end

"""
    active_bounds!(x,ℓ,u,fix_vars;atol)

Identifies the active components of `x`, i.e. lying at one of their bounds `ℓ` or `u`, using tolerance `atol` (default is `sqrt(eps)`).

This information is encoded in the `BitVector` `fix_vars`. The latter is modified in place and the components relative to already active bounds
are not modified.
"""
function active_bounds!(
    x::Vector,
    x_low::Vector,
    x_upp::Vector,
    fix_vars::BitVector;
    atol::Float64=sqrt(eps(Float64))) 

    for i in axes(fix_vars,1)
        fix_vars[i] = fix_vars[i] || (x[i] <= atol+x_low[i]) || (x_upp[i]-atol <= x[i])
    end
    return
end
"""
    active_bounds_step(bounds, x, s, Δ; atol)

Identifies the bounds that become active when taking the step `s` from `x` in the intersection of the feasible domain and the infinite norm trust region, encoded by `Δ`.
Returns the corresponding list of indices.

Uses the tolerance `atol`.
"""
function active_bounds_step(
    x::Vector,
    s::Vector,
    x_low::Vector,
    x_upp::Vector,
    delta::Float64;
    atol::Float64 = sqrt(eps(Float64))) 

    s_l = (t -> max(t,-delta)).(x_low-x)
    s_u = (t -> min(t,delta)).(x_upp-x)
    at_bound = BitVector(undef,size(x,1))

    for i in axes(s,1)
        at_bound[i] = (s[i]-s_l[i] <= atol) || (s_u[i]-s[i] <= atol)
    end

    active = findall(at_bound)
    return active
end

"""
    step_bounds(x,ℓ,u,Δ)

Computes and returns the lower and upper bounds for a step `s` that must satisfy `{ℓ ≤ x+s ≤ u, ||s|| ≤ Δ}` 
where `||.||` denotes the `∞`-norm.
"""
function step_bounds(
    x::Vector,
    x_low::Vector,
    x_upp::Vector,
    delta::Float64) 

    s_low = (t -> max(-delta, t)).(x_low-x)
    s_upp = (t -> min(delta,t)).(x_upp-x)

    return s_low, s_upp
end

"""
    add_active!(bounds, indx)

Sets the constraints in `indx` as active.
"""
function add_active!(
        bounds::BoundConstraints,
        indx::Vector{Int}) 

    bounds.fixvars[indx] .= true
    return
end
