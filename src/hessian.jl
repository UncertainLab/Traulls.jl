#=
    hessian.jl

Common interface of the augmented Lagrangian Hessian approximations (selection, update,
reset and products with vectors), Gauss-Newton approximation, dense secant approximations
and hybrid switching wrapper.

Author(s): Pierre Borie
=#

"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation (second-order terms set to 0)
- `sr1`: second-order terms updated by a SR1 formula
- `bfgs`: second-order terms updated by BFGS formula
- `hybrid_bfgs`: second-order terms updated by BFGS formula and used if non zero residuals
- `hybrid_sr1`: second-order terms updated by SR1 formula and used if non zero residuals
- `limited_sr1`: second-order terms approximated by a limited memory hessian and updated
  with SR1 formula
"""
@enum HessianApprox begin
    gn
    sr1
    bfgs
    hybrid_bfgs
    hybrid_sr1
    limited_sr1
end

const dict_hessians = Dict(:gn => gn,
                           :sr1 => sr1,
                           :bfgs => bfgs,
                           :hybrid_bfgs => hybrid_bfgs,
                           :hybrid_sr1 => hybrid_sr1,
                           :limited_sr1 => limited_sr1)

"""
    SecantHessian{T} <: ALHessian{T}

Abstract type of the approximations of the augmented Lagrangian Hessian of the form
`H = JᵀJ + μCᵀC + S`, where `J` and `C` are available first order quantities, `μ` is the
penalty parameter and `S` approximates the second order terms of the true Hessian.

Matrix `S` is updated iteratively from the pairs `(s, ŷ)` of the structured secant equation
`S₊s = ŷ`, where `s` is the accepted step and the right-hand side `ŷ` is defined by first
order quantities (see [`update_hessian!`](@ref)).

Subtypes must have the fields `J`, `C`, `mu`, `step`, `secant_rhs` and `temp` and implement
the methods

- `second_order_mul!(Hv, H, v, α)`: computes `Hv ← Hv + αSv`;
- `second_order_secant_update!(H)`: updates `S` from the pair stored in the fields `step`
  and `secant_rhs`;
- `reset_second_order!(H)`: sets `S` back to its initial value.

The Hessian-vector products, the update and the reset of the approximation are common to
all the subtypes.
"""
abstract type SecantHessian{T} <: ALHessian{T} end

"""
    SecantFormula

Abstract type of the marker types selecting the update formula of the second order terms of
a [`DenseSecantHessian`](@ref).
"""
abstract type SecantFormula end

"""
    GN <: ALHessian

Mutable structure representing the Gauss-Newton approximation of the Augmented
Lagrangian Hessian

# Attributes

- `J`: Jacobian of the residuals
- `C`: Jacobian of the nonlinear constraints
- `μ`: penalty parameter
- `temp`: buffer vector to avoid reallocations for intermediate
  quantities involved when computing matrix-vector products

The resulting approximation is `H = JᵀJ + μCᵀC`.
"""
mutable struct GN{T<:Real} <: ALHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    mu::T
    temp::AbstractVector{T}
end

"""
    GN(J,C,μ)

Constructor method for the [`GN`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the buffer
vector to zero.

# Arguments

- `J`: Jacobian matrix of the residuals
- `C`: Jacobian matrix of the nonlinear equality constraints
- `μ`: Penalty parameter
"""
function GN(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where T

    m = size(J,1)
    p = size(C,1)
    return GN(copy(J),copy(C),mu,zeros(T, max(m,p)))
end

"""
    DenseSecantHessian{T, F} <: SecantHessian{T}

Mutable structure encoding a secant approximation of the augmented Lagrangian Hessian whose
second order terms `S` are stored as a dense matrix and updated by the formula selected by
the marker type `F <: SecantFormula`.

The aliases [`SR1`](@ref) and [`BFGS`](@ref) correspond to the structured SR1 and BFGS
updates.

# Attributes

- `J`: Jacobian of the residuals
- `C`: Jacobian of the nonlinear constraints
- `S`: approximation of the second order terms of the true Hessian
- `mu`: penalty parameter
- `step`: step of the current iteration
- `secant_rhs`: right-hand side of the structured secant equation
- `nupdates`: number of updates applied to `S` since the last reset
- `temp`: buffer vector to avoid reallocations for intermediate quantities involved when
  computing matrix-vector products
"""
mutable struct DenseSecantHessian{T<:Real, F<:SecantFormula} <: SecantHessian{T}
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    S::AbstractMatrix{T}
    mu::T
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}
    nupdates::Int
    temp::AbstractVector{T}
end

"""
    DenseSecantHessian(F, J, C, μ)

Constructor method for the [`DenseSecantHessian`](@ref) structure with update formula `F`.

Takes jacobians and a penalty parameter as inputs. The second order terms are set to their
initial value (see [`reset_second_order!`](@ref)) and the other attributes to `0`.

# Arguments

- `F`: marker type of the update formula, e.g. `SR1Formula` or `BFGSFormula`
- `J`: Jacobian matrix of the residuals
- `C`: Jacobian matrix of the nonlinear constraints
- `μ`: Penalty parameter
"""
function DenseSecantHessian(
    ::Type{F},
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T) where {F<:SecantFormula, T}

    (m, n) = size(J)
    p = size(C, 1)

    H = DenseSecantHessian{T, F}(copy(J),
                                 copy(C),
                                 zeros(T, n, n),
                                 mu,
                                 zeros(T, n),
                                 zeros(T, n),
                                 0,
                                 zeros(T, max(n, m, p)))
    reset_second_order!(H)
    return H
end

"""
    Hybrid{T, H} <: ALHessian{T}

Mutable structure encoding the hybrid switching between the Gauss-Newton approximation and
the secant approximation `approx` of type `H <: SecantHessian`.

The second order terms of `approx` are only used when the problem is detected to have
non-small residuals. This detection relies on a heuristic based on the relative reduction of
the augmented Lagrangian objective over the last step (see [`update_hessian!`](@ref)). When
the residuals are deemed small, the approximation reduces to the Gauss-Newton one
`H = JᵀJ + μCᵀC`, but the second order terms keep being updated.

The aliases [`HybridSR1`](@ref) and [`HybridBFGS`](@ref) correspond to the hybrid
structured SR1 and BFGS approximations.

# Attributes

- `approx`: secant approximation, which stores the first and second order terms
- `small_res`: `true` if the residuals are deemed small, in which case the second order
  terms are not used in the Hessian-vector products
"""
mutable struct Hybrid{T, H<:SecantHessian{T}} <: ALHessian{T}
    approx::H
    small_res::Bool
end

"""
    Hybrid(approx)

Constructor method for the [`Hybrid`](@ref) structure. The residuals are initially not
considered small, i.e. `small_res` is set to `false`.
"""
Hybrid(approx::SecantHessian{T}) where T = Hybrid{T, typeof(approx)}(approx, false)

#= Products with vectors =#

"""
    first_order_mul!(Hv, H, v)

Compute the product `Hv = (JᵀJ + μCᵀC)v` of the first order terms of the Hessian
approximation `H` with vector `v` without doing matrix-matrix multiplications. The field
`temp` of `H` is used as a buffer for the intermediate products `Jv` and `Cv`.
"""
function first_order_mul!(Hv::AbstractVector{T}, H::ALHessian{T}, v::AbstractVector{T}) where T

    m = size(H.J, 1)
    p = size(H.C, 1)

    # JᵀJv term
    temp_Jv = view(H.temp, 1:m)
    mul!(temp_Jv, H.J, v) # form Jv
    mul!(Hv, H.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(H.temp, 1:p)
    mul!(temp_Cv, H.C, v) # form Cv
    mul!(Hv, H.C', temp_Cv, H.mu, 1) # Hv ← Hv + μCᵀCv

    return
end

"""
    mul!(Hv, H::ALHessian, v)

Overload the 3-argument `mul!` method to the Hessian approximations to compute the
Hessian-vector product `Hv` without doing matrix-matrix multiplications.

- For [`GN`](@ref), the product is `(JᵀJ + μCᵀC)v`.
- For a [`SecantHessian`](@ref), the product is `(JᵀJ + μCᵀC + S)v`.
- For a [`Hybrid`](@ref) approximation, `Sv` is added only if the attribute `small_res` is
  `false`.

The result is stored in `Hv`.
"""
function mul!(Hv::AbstractVector{T}, H::GN{T}, v::AbstractVector{T}) where T
    first_order_mul!(Hv, H, v)
    return
end

function mul!(Hv::AbstractVector{T}, H::SecantHessian{T}, v::AbstractVector{T}) where T
    first_order_mul!(Hv, H, v)
    second_order_mul!(Hv, H, v, one(T)) # Hv ← Hv + Sv
    return
end

function mul!(Hv::AbstractVector{T}, H::Hybrid{T}, v::AbstractVector{T}) where T
    first_order_mul!(Hv, H.approx, v)

    # If non zero residuals, Hv ← Hv + Sv
    !H.small_res && second_order_mul!(Hv, H.approx, v, one(T))
    return
end

"""
    Base.:*(H::ALHessian, v)

Overload the `*` operator to the Hessian approximations in order to avoid matrix-matrix
multiplications.
"""
function Base.:*(H::ALHessian{T}, v::AbstractVector{T}) where T
    Hv = Vector{T}(undef, size(v, 1))
    mul!(Hv, H, v)

    return Hv
end

"""
    second_order_mul!(Hv, H::DenseSecantHessian, v, α)

Compute `Hv ← Hv + αSv`, where `S` are the second order terms of `H`.
"""
function second_order_mul!(
    Hv::AbstractVector{T},
    H::DenseSecantHessian{T},
    v::AbstractVector{T},
    alpha::T) where T

    mul!(Hv, H.S, v, alpha, 1)
    return
end

#= Update and reset =#

"""
    update_jacobians!(M, M₊)

Replace the elements of matrix `M` by elements of matrix `M₊`.
The methods are specified the distinguish the case where `M` and `M₊` are `SparseMatrix`
with the same sparsity pattern.
"""
function update_jacobians!(hess_op::ALHessian, J_new::Matrix, C_new::Matrix)
    hess_op.J .= J_new
    hess_op.C .= C_new
    return
end

function update_jacobians!(hess_op::ALHessian, J_new::SparseMatrixCSC, C_new::SparseMatrixCSC)
    hess_op.J.nzval .= J_new.nzval
    hess_op.C.nzval .= C_new.nzval
    return
end

"""
    update_hessian!(H, J₊, C₊, r₊, c₊, f₊, f, g, y, s)

Update the Hessian approximation `H` after an accepted step `s`. All the approximations
share this signature.

- For [`GN`](@ref), the `J` and `C` attributes are replaced by, respectively, `J₊` and `C₊`.
  The other arguments are not used and can be omitted.
- For a [`SecantHessian`](@ref), the right-hand side of the structured secant equation

  `ŷ = [J₊ - J]ᵀr₊ + [C₊ - C]ᵀ[y + μc₊] = g - Jᵀr₊ - Cᵀ[y + μc₊]`

  is formed with the Jacobians `J` and `C` stored in `H`, then the second order terms are
  updated from the pair `(s, ŷ)` (see [`second_order_secant_update!`](@ref)) and the
  Jacobians are replaced by `J₊` and `C₊`. The objective values `f₊` and `f` are not used.
- For a [`Hybrid`](@ref) approximation, the small residuals heuristic is evaluated first:
  the residuals are deemed small if the relative reduction of the augmented Lagrangian
  objective satisfies

  `f - f₊ + δ > ε f,`

  where `ε = 0.1` (value recommended by Tjoa and Biegler[^TB91]) and `δ` is a small safeguard
  proportional to the machine precision. The result is stored in attribute `small_res`.
  The underlying secant approximation is then updated as above.

# Arguments

- `H`: Hessian approximation to update
- `J₊`: Jacobian of the residuals at the new point
- `C₊`: Jacobian of the nonlinear constraints at the new point
- `r₊`: residuals at the new point
- `c₊`: nonlinear constraints at the new point
- `f₊`: augmented Lagrangian objective value at the new point
- `f`: augmented Lagrangian objective value at the previous point
- `g`: gradient of the augmented Lagrangian at the new point
- `y`: Lagrange multipliers
- `s`: accepted step

# On return

- argument `H` modified in place to store the updated Hessian approximation

[^TB91]: I.B. Tjoa and L.T. Biegler, *Simultaneous solution and optimization strategies 
for parameter estimation of differential-algebraic equation systems*, Industrial & 
Engineering Chemistry Research, vol. 30(2), pp. 376-385, 1991.
"""
function update_hessian!(
    H::GN{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    _...) where T

    update_jacobians!(H, J_new, C_new)
    return
end

function update_hessian!(
    H::SecantHessian{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T},
    cx_new::AbstractVector{T},
    ::T,
    ::T,
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T}) where T

    # Secant equation right-hand side ŷ ← g - Jᵀr₊ - Cᵀ(y + μc₊)
    rhs = H.secant_rhs
    rhs .= g
    mul!(rhs, H.J', rx_new, -1, 1)
    w = view(H.temp, 1:size(cx_new, 1))
    w .= y .+ H.mu .* cx_new
    mul!(rhs, H.C', w, -1, 1)
    H.step .= s

    # Update second order terms
    second_order_secant_update!(H)

    # Update first order terms
    update_jacobians!(H, J_new, C_new)

    return
end

function update_hessian!(
    H::Hybrid{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx_new::AbstractVector{T},
    cx_new::AbstractVector{T},
    fx_new::T,
    fx_prev::T,
    g::AbstractVector{T},
    y::AbstractVector{T},
    s::AbstractVector{T}) where T

    # Constants
    eps_small_res = T(1//10) # Tjoa Biegler recommended value
    eps_red = 10 * eps(T)

    # Evaluate small residuals heuristic
    delta_red = eps_red * max(1, abs(fx_prev))
    H.small_res = (fx_prev - fx_new + delta_red) > eps_small_res * fx_prev

    # Update the underlying secant approximation
    update_hessian!(H.approx, J_new, C_new, rx_new, cx_new, fx_new, fx_prev, g, y, s)

    return
end

"""
    reset_hessian!(H, J₀, C₀, μ₀)

Reset the Hessian approximation `H` at the start of a new outer iteration, by setting the
`J`, `C` and `mu` attributes to, respectively, `J₀`, `C₀` and `μ₀`.

The second order terms are set back to their initial value (see
[`reset_second_order!`](@ref)). For a [`Hybrid`](@ref) approximation, the residuals are no
longer considered small (`small_res` is set to `false`).
"""
function reset_hessian!(
    H::ALHessian{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    update_jacobians!(H, J0, C0)
    H.mu = mu0
    reset_second_order!(H)
    return
end

function reset_hessian!(
    H::Hybrid{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    reset_hessian!(H.approx, J0, C0, mu0)
    H.small_res = false
    return
end

"""
    reset_second_order!(H)

Set the second order terms of the Hessian approximation `H` back to their initial value.

For a [`DenseSecantHessian`](@ref), the initial value depends on the update formula (see
[`initialize_second_order!`](@ref)) and the update counter `nupdates` is set to `0`.
"""
reset_second_order!(::GN) = nothing

function reset_second_order!(H::DenseSecantHessian)
    initialize_second_order!(H)
    H.nupdates = 0
    return
end

# Source files for the secant approximations
include("sr1.jl")
include("bfgs.jl")
include("lsr1.jl")
