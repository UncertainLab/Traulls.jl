#=
    bfgs.jl

Structured BFGS and hybrid BFGS approximations of the augmented Lagrangian Hessian, where
the second order terms are updated with a BFGS formula based on an NL2SOL-style secant
equation.

Author(s): Pierre Borie
=#

"""
    BFGSFormula <: SecantFormula

Marker type selecting the structured BFGS update of the second order terms of a
[`DenseSecantHessian`](@ref).
"""
struct BFGSFormula <: SecantFormula end

"""
    BFGS{T}

Alias of `DenseSecantHessian{T, BFGSFormula}`, the BFGS approximation of the augmented
Lagrangian Hessian `H = JᵀJ + μCᵀC + S`.

Matrix `S` is initialized to the identity, rescaled at the first update, and updated 
iteratively by a BFGS formula derived from a NL2SOL style structured secant equation 
`Ss = y` where `s` is a step and right-hand side `y` is defined by first order quantities
(see [`second_order_secant_update!`](@ref)).
"""
const BFGS{T} = DenseSecantHessian{T, BFGSFormula}

"""
    HybridBFGS{T}

Alias of `Hybrid{T, BFGS{T}}`, the hybrid switching between the Gauss-Newton and the
[`BFGS`](@ref) approximations of the augmented Lagrangian Hessian.
"""
const HybridBFGS{T} = Hybrid{T, BFGS{T}}

# Constructor for the BFGS approximation
BFGS(J::AbstractMatrix{T}, C::AbstractMatrix{T}, mu::T) where T =
    DenseSecantHessian(BFGSFormula, J, C, mu)

# Constructor for the hybrid BFGS approximation
# The `small_res` parameter is set to `false`
HybridBFGS(J::AbstractMatrix{T}, C::AbstractMatrix{T}, mu::T) where T =
    Hybrid(BFGS(J, C, mu))

# Second order terms are initialized with identity
function initialize_second_order!(H::BFGS{T}) where T
    H.S .= zero(T)
    for i in axes(H.S, 1)
        H.S[i, i] = one(T)
    end
    return
end

# Apply the `BFGS` secant update to the second order terms approximation
# TODO: try out the damped Powell update
function second_order_secant_update!(bfgs_op::BFGS{T}) where T

    # Tolerance for skipping update
    eps_skip = T(1e-7)
    y = bfgs_op.secant_rhs
    s = bfgs_op.step

    sty = dot(s, y)

    # Update second order terms it sufficient curvature
    if sty >= eps_skip * (1 + norm(s) * norm(y))

        # Rescale initial approximation at first update
        if bfgs_op.nupdates == 0
            scaling_factor = sty * (1 / dot(s, s))
            bfgs_op.S .= T(0)
            for i in axes(bfgs_op.S, 1)
                bfgs_op.S[i, i] = scaling_factor
            end
        end

        # Buffer to store matrix-vector product
        Ss = view(bfgs_op.temp, 1:size(s, 1))
        mul!(Ss, bfgs_op.S, s)

        # Apply BFGS update
        mul!(bfgs_op.S, y, y', 1/sty, 1)           # S ← S + yyᵀ/sᵀy
        mul!(bfgs_op.S, Ss, Ss', -1/dot(s, Ss), 1) # S ← S - SssᵀS / sᵀSs
        bfgs_op.nupdates += 1

    end

    return
end
