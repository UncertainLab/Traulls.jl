# Problem 5.4 from Luksan and Vlcek collection (1999)
# Chained Cragg-Levy function with tridiagonal constraints

function lv504(n::Int=100)

    @assert n >= 4 "n must be greater or equal than 4"
    if n % 2 != 0
        "n must be even, rounding up"
        n += 1
    end


    N = div(n, 2) - 1
    m = 5N  
    p = n-2

    # Residuals

    function r!(rx, x)

        N = div(size(x,1),2) - 1

        for i=1:N
            rx[i] = (exp(x[2i-1]) - x[2i])^2
            rx[N+i] = 10*(x[2i] - x[2i+1])^3
            rx[2N+i] = tan(x[2i+1] - x[2i+2])^2
            rx[3N+i] = x[2i-1]^4
            rx[4N+i] = x[2i+2]-1
        end

        return
    end

    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

    # Sparse version of the Jacobian
    function jr!(J, x)
        N = div(size(x,1),2) - 1

        for i=1:N
            buff = exp(x[2i-1]) - x[2i]
            J[i, 2i-1] = 2 * exp(x[2i-1]) * buff
            J[i, 2i] = -2 * buff

            buff = (x[2i] - x[2i+1])^2
            J[N+i, 2i] = 30 * buff
            J[N+i, 2i+1] = -30 * buff

            tan_term = tan(x[2i+1] - x[2i+2])
            buff =  (1 + tan_term^2) * tan_term
            J[2N+i, 2i+1] = 2 * buff
            J[2N+i, 2i+2] = -2 * buff

            J[3N+i, 2i-1] = 4 * x[2i-1]^3
            J[4N+i, 2i+2] = 1
        end
        return
    end

    # Sparsity pattern
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i=1:N
        append!(jr_nzrows, [i, i]); append!(jr_nzcols, [2i-1, 2i])
        append!(jr_nzrows, [N+i, N+i]); append!(jr_nzcols, [2i, 2i+1])
        append!(jr_nzrows, [2N+i, 2N+i]); append!(jr_nzcols, [2i+1, 2i+2])
        append!(jr_nzrows, [3N+i]); append!(jr_nzcols, [2i-1])
        append!(jr_nzrows, [4N+i]); append!(jr_nzcols, [2i+2])
    end

    # Constraints

    function c!(cx, x)
        cx .= [8 * x[k+1] * (x[k+1]^2 - x[k]) - 2(1 - x[k+1]) + 4(x[k+1] - x[k+2]^2)
               for k=1:size(x,1)-2]
        return
    end

    jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(p), x)

    # Sparse version of the Jacobian
    function jc!(C, x)
        n = size(x, 1)
        for k = 1:n-2
            C[k, k] = -8 * x[k+1]
            C[k, k+1] = 24 * x[k+1]^2 - 8 * x[k] + 6
            C[k, k+2] = -8 * x[k+2]
        end
        return
    end

    # Sparsity pattern
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for k=1:n-2
        append!(jc_nzrows, [k, k, k]); append!(jc_nzcols, [k, k+1, k+2])
    end
    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point

    x = [mod(i,4) == 1 ? 1.0 : 2.0 for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))

    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
