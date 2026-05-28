# Structure encoding a limite memory SR1 (L-SR1) approximation in compact representation
# format
# TODO: Think about one field for middle matrix instead of two
# When updating, use it to form the middle matrix
# Modify in place this same field to store the LU decomposition
mutable struct LSR1{T <: Real} <: ALHessian{T}
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

    # Buffers for intermediate computations
    temp::AbstractVector{T}
    secant_rhs::AbstractVector{T}
end

# Constructor
function LSR1(
    J::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mu::T;
    mem = 14) where T

    (m,n) = size(J)
    p = size(C, 1)

    LSR1(copy(J),
         copy(C),
         mu,
         max(mem, 1),
         0,
         [zeros(T, n) for _ = 1:mem],
         [zeros(T, n) for _ = 1:mem],
         T(1),
         zeros(T, mem, mem),
         lu(ones(T, 1, 1)),
         zeros(T, max(n, m, p)),
         zeros(T, n))
end

# 3-argument `mul!` overload
function mul!(Hv::AbstractVector, lsr1::LSR1, v::AbstractVector)

    # Product with first order terms
    m = size(lsr1.J, 1)
    p = size(lsr1.C, 1)
    n = size(v, 1)

    # JᵀJv term
    temp_Jv = view(lsr1.temp, 1:m)
    mul!(temp_Jv, lsr1.J, v) # form Jv
    mul!(Hv, lsr1.J', temp_Jv, 1, 0) # Hv ← JᵀJv

    # μCᵀCv term
    temp_Cv = view(lsr1.temp, 1:p)
    mul!(temp_Cv, lsr1.C, v) # form Cv
    mul!(Hv, lsr1.C', temp_Cv, lsr1.mu, 1) # Hv ← Hv + μCᵀCv

    # Bv term
    temp_Bv = view(lsr1.temp, 1:n)
    second_order_product!(temp_Bv, lsr1, v)
    Hv .+= temp_Bv

    return
end

# Computes the product of the second order terms approximated by LSR1
# TODO: Avoid allocation for auxiliary vector, suspect something to go wrong if the
# buffer field is used for both `res` and used to define `w`
function second_order_product!(
    res::AbstractVector{T},
    lsr1::LSR1{T},
    v::AbstractVector{T}) where T

    k = lsr1.stored_pairs
    sigma = lsr1.scaling_factor

    # Accumulated information terms
    if k > 0
        w = zeros(k)

        S, Y = lsr1.stored_steps, lsr1.stored_rhs

        # w ← (Y - σS)ᵀv
        for i=1:k
            w[i] = dot(Y[i], v) - sigma * dot(S[i], v)
        end

        # w ← M⁻¹w
        ldiv!(lsr1.lu_middle_mat, w)

        # res ← (Y - σS)w
        for i = 1:k
            res .+= (Y[i] .- sigma .* S[i]) .* w[i]
        end

    end

    # Initial term
    res .+= v .* sigma

    return
end

# Update of the approximation
function update_hessian!(
    lsr1::LSR1{T},
    J_new::AbstractMatrix{T},
    C_new::AbstractMatrix{T},
    rx::Vector{T},
    cx::Vector{T},
    g::Vector{T},
    lambda::Vector{T},
    s::Vector{T}) where T

    # Tolerance for the skipping update safeguard
    eps_skip = T(1e-8)

    n = size(s, 1)
    # Update Jacobians
    update_jacobians!(lsr1, J_new, C_new)

    # Form the secant equations components
    y = g - lsr1.J' * rx .- lsr1.C' * (lambda .+ cx .* lsr1.mu)

    # Form y - Bs
    ymBs = view(lsr1.temp, 1:n)
    second_order_product!(ymBs, lsr1, s) # ymBs ← Bs
    ymBs .= y .- ymBs                    # ymBs ← y - Bs
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

    # Feed remaining elements exploiting symmetry
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
# Reset the fields of the approximation at the start of a new outer iteration
function reset_hessian!(
    H::LSR1{T},
    J0::AbstractMatrix{T},
    C0::AbstractMatrix{T},
    mu::T) where T

    n = size(J0, 2)
    zero_T = zero(T)
    one_T = one(T)
    mem = H.memory

    update_jacobians!(H, J0, C0)
    H.mu = mu

    H.stored_pairs = 0
    H.stored_steps = [zeros(T, n) for _ = 1:mem]
    H.stored_rhs = [zeros(T, n) for _ = 1:mem]
    H.scaling_factor = one_T
    H.middle_mat .= zero_T
    H.lu_middle_mat = lu(ones(T, 1,1))

end
