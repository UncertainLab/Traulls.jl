#=
    lsr1.jl

Limited memory SR1 (L-SR1) approximation of the augmented Lagrangian Hessian in compact
representation.

Author(s): Pierre Borie
=#

# Structure encoding a limited memory SR1 (L-SR1) approximation in compact representation
# format
# TODO: Think about one field for middle matrix instead of two
# When updating, use it to form the middle matrix
# Modify in place this same field to store the LU decomposition
mutable struct LSR1{T <: Real} <: SecantHessian{T}
    # First order fields
    J::AbstractMatrix{T}
    C::AbstractMatrix{T}
    mu::T

    # Compact representation fields
    memory::Int
    stored_pairs::Int
    stored_steps::Vector{Vector{T}}
    stored_rhs::Vector{Vector{T}}
    scaling_factor::T
    middle_mat::AbstractMatrix{T}
    lu_middle_mat::Factorization

    # Secant equation components
    step::AbstractVector{T}
    secant_rhs::AbstractVector{T}

    # Buffer for intermediate computations
    temp::AbstractVector{T}
end

# Constructor
function LSR1(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T;
    mem = 14) where T

    (m,n) = size(J)
    p = size(C, 1)
    mem = max(mem, 1)

    LSR1(copy(J),
         copy(C),
         mu,
         mem,
         0,
         [zeros(T, n) for _ = 1:mem],
         [zeros(T, n) for _ = 1:mem],
         T(1),
         zeros(T, mem, mem),
         lu(ones(T, 1, 1)),
         zeros(T, n),
         zeros(T, n),
         zeros(T, max(n, m, p)))
end

# Computes `Hv ← Hv + αBv`, where `B` is the L-SR1 approximation of the second order terms
# TODO: Avoid allocation for auxiliary vector `w`
function second_order_mul!(
    Hv::AbstractVector{T},
    lsr1::LSR1{T},
    v::AbstractVector{T},
    alpha::T) where T

    k = lsr1.stored_pairs
    sigma = lsr1.scaling_factor

    # Accumulated information terms
    if k > 0
        w = zeros(T, k)

        S, Y = lsr1.stored_steps, lsr1.stored_rhs

        # w ← (Y - σS)ᵀv
        for i=1:k
            w[i] = dot(Y[i], v) - sigma * dot(S[i], v)
        end

        # w ← M⁻¹w
        ldiv!(lsr1.lu_middle_mat, w)

        # Hv ← Hv + α(Y - σS)w
        for i = 1:k
            Hv .+= alpha .* w[i] .* (Y[i] .- sigma .* S[i])
        end

    end

    # Initial term
    Hv .+= (alpha * sigma) .* v

    return
end

# Add the pair `(s, y)` stored in `lsr1` to the approximation
# The pair is skipped if the denominator (y - Bs)ᵀs is too small
function second_order_secant_update!(lsr1::LSR1{T}) where T

    # Tolerance for the skipping update safeguard
    eps_skip = T(1e-8)

    y = lsr1.secant_rhs
    s = lsr1.step

    # Form y - Bs
    ymBs = view(lsr1.temp, 1:size(s, 1))
    ymBs .= y
    second_order_mul!(ymBs, lsr1, s, -one(T)) # ymBs ← y - Bs
    denom = dot(s, ymBs)

    # Add pair if denominator (y - Bs)ᵀs not zero
    if abs(denom) > eps_skip * (1 + norm(s) * norm(ymBs))
        add_pair!(lsr1, s, y)
        update_compact_representation!(lsr1)
    end

    return
end

# Add an accepted pair to the L-SR1 approximation structure and update the scaling parameter
# TODO: Think about a pattern to avoid cycling when adding / removing a pair
function add_pair!(lsr1::LSR1{T}, s::AbstractVector{T}, y::AbstractVector{T}) where T

    eps_curv = sqrt(eps(T))
    nstored = lsr1.stored_pairs
    m = lsr1.memory

    # If the number of pairs is below the limit, simply add
    if nstored < m
        i_insert = nstored + 1
        lsr1.stored_steps[i_insert] .= s
        lsr1.stored_rhs[i_insert] .= y

        lsr1.stored_pairs += 1
    else
        # Remove the oldest pairs and move the remaing one at previous position
        for k = 1:m-1
            lsr1.stored_steps[k] .= lsr1.stored_steps[k+1]
            lsr1.stored_rhs[k] .= lsr1.stored_rhs[k+1]
        end

        # Insert the new pair at final position
        lsr1.stored_steps[m] .= s
        lsr1.stored_rhs[m] .= y
    end

    # If the curvature is positive enough, update the scaling parameter, if not, the
    # previous one is maintained
    yy = dot(y,y)
    sty = dot(s,y)

    sigma = yy * (1 / sty) # σ ← sᵀy / yᵀy / sᵀy
    if sty > eps_curv * (1 + norm(s) * sqrt(yy))
        lsr1.scaling_factor = sigma
    end

    return
end

# Update the components of the compact representation of the L-SR1 approximation
# Modifies the middle matrix and its LU factorization components
# TODO: Modifies the update operations in order the avoid unnessary recomputations
function update_compact_representation!(lsr1::LSR1)

    m = lsr1.stored_pairs
    M = view(lsr1.middle_mat, 1:m, 1:m)
    S, Y = lsr1.stored_steps, lsr1.stored_rhs
    sigma = lsr1.scaling_factor

    # Feed diagonal
    for i = 1:m
        si = S[i]
        M[i, i] = dot(si, Y[i]) - sigma * dot(si, si)
    end

    # Feed remaining elements
    for i = 2:m
        for j = 1:i-1
            si = S[i]
            elem = dot(si, Y[j]) - sigma * dot(si, S[j])
            M[i,j] = elem
            M[j,i] = elem
        end
    end

    # Update LU factorisation of the middle matrix
    lsr1.lu_middle_mat = lu(M)

    return
end

# Remove the stored pairs at the start of a new outer iteration
function reset_second_order!(H::LSR1{T}) where T

    H.stored_pairs = 0
    H.scaling_factor = one(T)
    H.middle_mat .= zero(T)
    H.lu_middle_mat = lu(ones(T, 1,1))

    return
end
