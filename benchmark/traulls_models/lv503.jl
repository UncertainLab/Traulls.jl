# Problem 5.3 from Luksan & Vlsek collection (1999)
# Chained Powell singular function with simplified trigonometric-exponential constraints

function lv503(n::Int=100)

    @assert n  >= 2 "n must ge greater or equal than 2"
    if n % 2 != 0
        @warn "n must be even, rounding up"
        n += 1
    end

    N = div(n, 2) - 1
    m = 4N
    p = 2

    # Residuals

    function r!(rx, x)
        N = div(size(x,1),2) - 1
        sfive = sqrt(5)
        sten = sqrt(10)
        for i=1:N
            rx[i] = x[2i-1] + 10*x[2i]
            rx[N+i] = sfive*(x[2i+1] - x[2i+2])
            rx[2N+i] = (x[2i] - 2*x[2i+1])^2
            rx[3N+i] = sten*(x[2i-1] - x[2i+2])^2
        end
        return
    end

    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

    # Sparse version of the jacobian
    function jr!(J, x)
        N = div(size(x, 1), 2) - 1
        sfive = sqrt(5)
        sten = sqrt(10)
        for i=1:N
            J[i, 2i-1] = 1
            J[i, 2i] = 10
            J[N+i, 2i+1] = sfive
            J[N+i, 2i+2] = -sfive
            J[2N+i, 2i] = 2 * x[2i] - 4 * x[2i+1]
            J[2N+i, 2i+1] = -4 * x[2i] + 8 * x[2i+1]
            J[3N+i, 2i-1] = 2 * sten * (x[2i-1] - x[2i+2])
            J[3N+i, 2i+2] = -2 * sten * (x[2i-1] - x[2i+2])
        end
        return
    end

    # Sparsity pattern
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i=1:N
        append!(jr_nzrows, [i, i]); append!(jr_nzcols, [2i-1, 2i])
        append!(jr_nzrows, [N+i, N+i]); append!(jr_nzcols, [2i+1, 2i+2])
        append!(jr_nzrows, [2N+i, 2N+i]); append!(jr_nzcols, [2i, 2i+1])
        append!(jr_nzrows, [3N+i, 3N+i]); append!(jr_nzcols, [2i-1, 2i+2])
    end

    # Constraints

    function c!(cx, x)
        n = size(x,1)

        cx[1] = 3x[1]^3 + 2x[2] - 5 + sin(x[1]-x[2]) * sin(x[1]+x[2])
        cx[2] = 4x[n] - x[n-1]*exp(x[n-1]-x[n]) - 3

        return
    end

    jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(p), x)

    # Sparse version of the Jacobian
    function jc!(C, x)
        n = size(x, 1)

        C[1, 1] = 9 * x[1]^2 + cos(x[1] - x[2]) * sin(x[1] + x[2]) + sin(x[1] - x[2]) *
            cos(x[1] + x[2])

        C[1, 2] = 2 - cos(x[1] - x[2]) * sin(x[1] + x[2]) + sin(x[1] - x[2]) *
            cos(x[1] + x[2])

        C[2, n-1] = -exp(x[n-1] - x[n]) * (1 + x[n-1])
        C[2, n] = 4 + x[n-1] * exp(x[n-1] - x[n])

        return
    end

    # Sparsity pattern
    jc_nzrows = [1, 1, 2, 2]
    jc_nzcols = [1, 2, n-1, n]


    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point

    start_values = [1.0, 3, -1, 0]

    x = [start_values[mod(i,4) + 1] for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x,n,m,p,Val(:only_equalities))

    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    x_low, x_upp, x, n, m, p, Val(:only_equalities))
end
