"""
    HessianApprox

`Enum` type to caracterize the differente Hessian approximations used in our solver.

- `gn`: Gauss-Newton approximation 
- `sr1`: second-order terms updates by a SR1 formula
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
    GN <: ALHessian 

Mutable structure representing the Gauss-Newton approximation of the Augmented
Lagrangian Hessian

**Attributes**

* `J`: Jacobian of the residuals 

* `C`: Jacobian of the nonlinear constraints 

* `μ`: penalty parameter

* `temp`: buffer vector to avoid reallocations of the intermediate
quantities involved when computing matrix-vector products

The resulting approximation is `H = JᵀJ + μCᵀC`.
"""
mutable struct GN{T<:Real, MJ<:AbstractMatrix{T}, MC<:AbstractMatrix{T}} <: ALHessian{T}
    J::MJ
    C::MC
    mu::T
    temp::Vector{T}
end

"""
    GN(J,C,μ)

Constructor method for the [`GN`](@ref) structure.

Takes jacobians and a penalty parameter as inputs and initializes the buffer
vector to zero.

* Arguments

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
    return GN(copy(J),copy(C),mu,zeros(T,max(m,p)))
end

"""
    mul!(Hv, H::GN, v)

Overload the 3-argument `mul!` method to the type [`GN`](@ref) to compute
Hessian-vector without doing matrix-matrix multiplications.
"""

function mul!(Hv::Vector{T}, gn_op::GN{T}, v::Vector{T}) where T

    m = size(gn_op.J,1)
    p = size(gn_op.C,1)
    
    # Reset result values to make sure it is zero
    # Hv .= 0.0
    # JᵀJv term
    temp_Jv = view(gn_op.temp,1:m)
    mul!(temp_Jv, gn_op.J, v) # form Jv
    mul!(Hv, gn_op.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(gn_op.temp,1:p)
    mul!(temp_Cv, gn_op.C, v) # form Cv
    mul!(Hv, gn_op.C', temp_Cv, gn_op.mu, 1) # Hv ← Hv + μCᵀCv

    return
end



""" Base.:*(H::GN, v)

Overload the `*` operator to the type [`GN`](@ref) in order to avoid
matrix-matrix multiplication
"""
function Base.:*(H::GN{T}, v::Vector{T}) where T
    Hv = Vector{T}(undef,size(v,1))
    mul!(Hv,H,v)

    return Hv
end

"""
    update_jacobian(M, M₊)

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
end
"""
    update_hessian!(H, J₊, C₊)

Updates the Gauss-Newton Hessian approximation `H` by modifiying the `J` and `C`
attributes with, respectively `J₊` and `C₊`.
"""
function update_hessian!(
    H::GN{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T}) where T
    
    update_jacobians!(H, J_new, C_new)
    return
end

"""
    reset_hessian!(H,J₀,C₀,μ₀)

Reset the Gauss-Newton approximation `H` by setting the `J`, `C` and `mu`
attributes to, respectively, `J₀`, `C₀` and μ₀.
"""
function reset_hessian!(
    H::GN{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu0::T) where T

    H.J .= J0
    H.C .= C0
    H.mu = mu0
    return
end

# Source files for structured hessian approximations
include("sr1.jl")
include("bfgs.jl")
include("lsr1.jl")
