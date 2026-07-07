function lv513(n::Int=100)
    @assert n >= 5 "n must be ≥ 5"
    if n % 3 != 2
        @warn("n must equal 2 modulo 3, rounding up")
        n = div(n-2,3) * 3 + 5
    end

    N = div(n-2,3)
    m = 3N
    p = 2N

    # Residuals

    function r!(rx, x)
        N = div(size(x,1) - 2, 3)

        for i = 1:N
            rx[i] = x[3i-2] - 1
            rx[N+i] = x[3i-1] - x[3i]
            rx[2N+i] = (x[3i+1] - x[3i+2])^2
        end

        return
    end

    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

    # Sparse version of the Jacobian
    function jr!(J, x)
        n = size(x, 1)
        N = div(n - 2, 3)

        for i = 1:N
            J[i, 3i-2] = 1
            J[N+i, 3i-1] = 1
            J[N+i, 3i] = -1
            J[2N+i, 3i+1] = 2 * (x[3i+1] - x[3i+2])
            J[2N+i, 3i+2] = -2 * (x[3i+1] - x[3i+2])
        end

        return
    end

    # Sparsity pattern
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i=1:N
        push!(jr_nzrows, i); push!(jr_nzcols, 3i-2)
        append!(jr_nzrows, [N+i, N+i]); append!(jr_nzcols, [3i-1, 3i])
        append!(jr_nzrows, [2N+i, 2N+i]); append!(jr_nzcols, [3i+1, 3i+2])
    end
    # Constraints

    function c!(cx, x)
        N = div(size(x,1) - 2, 3)

        for k = 1:2N
            l = 3*div(k-1,2)

            if k % 2 == 1
                cx[k] = x[l+1] + x[l+2]^2 + x[l+3] + x[l+4] + x[l+5] - 5
            else
                cx[k] = x[l+3]^2 - 2 * (x[l+4] + x[l+5]) - 3
            end
        end

        return
    end

    jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(p), x)

    # Sparse version of the Jacobian
    function jc!(A, x)
        n = size(x, 1)
        N = div(n-2, 3)

        for k = 1:2N
            l = 3 * div(k-1, 2)
            if mod(k,2) == 1
                A[k, l+1] = 1
                A[k, l+2] = 2 * x[l+2]
                A[k, l+3] = 1
                A[k, l+4] = 1
                A[k, l+5] = 1
            else
                A[k, l+3] = 2 * x[l+3]
                A[k, l+4] = -2
                A[k, l+5] = -2
            end
        end
        return
    end

    # Sparsity pattern
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for k = 1:p
        l = 3 * div(k-1, 2)
        if mod(k,2) == 1
            append!(jc_nzrows, [k, k, k, k, k])
            append!(jc_nzcols, [l+1, l+2, l+3, l+4, l+5])
        else
            append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [l+3, l+4, l+5])
        end
    end

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point
    x0_values = [3.0, 5.0, -3.0]
    x = [x0_values[mod(i-1,3) + 1] for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))
    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
