function lv511(n::Int=100)

    @assert n >= 5 "n must be ≥ 5"

    if n % 3 != 2
        @warn "n must equal 2 modulo 3, rounding up"
        n = div(n-2,3)*3 + 5
    end

    N = div(n-2,3)
    m = 4N
    p = 2N

    # Residuals
    function r!(rx, x)
        N = div(size(x,1)-2,3)

        for i=1:N
            rx[i] = x[3i-2] - x[3i-1]
            rx[N+i] = x[3i] - 1
            rx[2N+i] = (x[3i+1] - 1)^2
            rx[3N+i] = (x[3i+2] - 1)^3
        end
        return
    end

    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

    # Sparse version of the Jacobian
    function jr!(J, x)
        n = size(x, 1)
        N = div(n-2, 3)

        for i = 1:N
            J[i, 3i-2] = 1
            J[i, 3i-1] = -1
            J[N+i, 3i] = 1
            J[2N+i, 3i+1] = 2 * (x[3i+1] - 1)
            J[3N+i, 3i+2] = 3 * (x[3i+2] - 1)^2
        end
        return
    end

    # Sparsity pattern
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i=1:N
        append!(jr_nzrows, [i, i]); append!(jr_nzcols, [3i-2, 3i-1])
        push!(jr_nzrows, N+i); push!(jr_nzcols, 3i)
        push!(jr_nzrows, 2N+i); push!(jr_nzcols, 3i+1)
        push!(jr_nzrows, 3N+i); push!(jr_nzcols, 3i+2)
    end

    # Constraints
    function c!(cx, x)
        n = size(x, 1)
        
        N = div(n-2,3)

        for k = 1:2N
            l = 3*div(k-1,2)
            if mod(k,2) == 1
                cx[k] = x[l+1]^2 * x[l+4] + sin(x[l+4]-x[l+5]) - 1
            else
                cx[k] = x[l+2] + x[l+3]^4*x[l+4]^2 - 2
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
                A[k, l+1] = 2 * x[l+1] * x[l+4]
                A[k, l+4] = x[l+1]^2 + cos(x[l+4] - x[l+5])
                A[k, l+5] = -cos(x[l+4] - x[l+5])
            else
                A[k, l+2] = 1
                A[k, l+3] = 4 * x[l+3]^3 * x[l+4]^2
                A[k, l+4] = 2 * x[l+3]^4 * x[l+4]
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
            append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [l+1, l+4, l+5])
        else
            append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [l+2, l+3, l+4])
        end
    end

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point
    x0_values = [2.0, 1.5, 0.5]

    x = [x0_values[mod(i-1,3)+1] for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))

    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
