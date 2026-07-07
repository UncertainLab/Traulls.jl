# Problem 5.12 from Lksan & Vlcek collection (1999)
# Chained HS47 problem

function lv512(n::Int=100)

    @assert n > 5 "n must be greater or equal than 5"

    if mod(n,4) != 1
        @warn("number of variables must equal 1 modulo 4, rounding up")
        n = 4 * div(n-1,4) + 5
    end


    N = div(n-1,4)
    m = 4N
    p = 3*N


    # Residuals
    function r!(rx, x)
        N = div(size(x,1)-1,4)

        for i=1:N
            rx[i] = x[4i-3] - x[4i-2]
            rx[N+i] = x[4i-2] - x[4i-1]
            rx[2N+i] = (x[4i-1] - x[4i])^2
            rx[3N+i] = (x[4i] - x[4i+1])^2
        end
        return
    end

    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)


    # Sparse version of the Jacobian
    function jr!(J, x)
        n = size(x, 1)
        N = div(n-1, 4)

        for i = 1:N
            J[i, 4i-3] = 1
            J[i, 4i-2] = -1
            J[N+i, 4i-2] = 1
            J[N+i, 4i-1] = -1
            J[2N+i, 4i-1] = 2 * (x[4i-1] - x[4i])
            J[2N+i, 4i] = -2 * (x[4i-1] - x[4i])
            J[3N+i, 4i] = 2 * (x[4i] - x[4i+1])
            J[3N+i, 4i+1] = -2 * (x[4i] - x[4i+1])
        end
        return
    end

     # Sparsity pattern
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i=1:N
        append!(jr_nzrows, [i, i]); append!(jr_nzcols, [4i-3, 4i-2])
        append!(jr_nzrows, [N+i, N+i]); append!(jr_nzcols, [4i-2, 4i-1])
        append!(jr_nzrows, [2N+i, 2N+i]); append!(jr_nzcols, [4i-1, 4i])
        append!(jr_nzrows, [3N+i, 3N+i]); append!(jr_nzcols, [4i, 4i+1])
    end

    # Constraints

    function c!(cx, x)
        N = div(size(x, 1) - 1, 4)

        for k = 1:3N
            l = 4 * div(k - 1, 3)
            if mod(k,3) == 1
                cx[k] = x[l+1] + x[l+2]^2 + x[l+3]^2 - 3
            elseif k % 3 == 2
                cx[k] = x[l+2] + x[l+3]^2 + x[l+4] - 1
            else
                cx[k] = x[l+1] * x[l+5] - 1
            end
        end
        return
    end

    jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(p), x)


    # Sparse version of the Jacobian
    function jc!(A, x)
        n = size(x, 1)
        N = div(n-1, 4)

        for k = 1:3N
            l = 4 * div(k-1, 3)
            if mod(k,3) == 1
                A[k, l+1] = 1
                A[k, l+2] = 2 * x[l+2]
                A[k, l+3] = 2 * x[l+3]
            elseif mod(k, 3) == 2
                A[k, l+2] = 1
                A[k, l+3] = 2 * x[l+3]
                A[k, l+4] = 1
            else
                A[k, l+1] = x[l+5]
                A[k, l+5] = x[l+1]
            end
        end
        return
    end

    # Sparsity pattern
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for k = 1:p
        l = 4 * div(k-1, 3)
        if mod(k, 3) == 1
            append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [l+1, l+2, l+3])
        elseif mod(k, 3) == 2
            append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [l+2, l+3, l+4])
        else
            append!(jc_nzrows, [k, k]); append!(jc_nzcols, [l+1, l+5])
        end
    end

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point
    x0_values = [2.0, 1.5, -1.0, 0.5]

    x = [x0_values[mod(i-1,4)+1] for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))

    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
