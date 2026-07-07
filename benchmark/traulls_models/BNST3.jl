# Example 3 from Biegler et al. (2000)
# Numerical Experience with a Reduced Hessian Method for Large Scale Constrained Optimization

function BNST3(n::Int=100)

    if n < 2
        @warn("BNST3: number of variables must be >= 2. Setting to 2")
        n = 2
    elseif n % 2 == 1
        @warn("BNST3: n must be even. Rounding up")
        n += 1
    end

    m = n
    p = div(n, 2)

    function r!(rx, x)
        rx .= x 
        return
    end

    function jac_r!(J, x)
       
        J .= 0
        for i in axes(x, 1)
            J[i, i] = 1.0
        end

        return 
    end

    # Sparse version of the jacobian 
    function jr!(J, x)
        one_T = one(eltype(x))
        for i in axes(x, 1)
            J[i,i] = one_T
        end 
    end

    # Sparsity pattern
    jr_nzrows = collect(1:m)
    jr_nzcols = collect(1:n)

    function c!(cx, x)

        n = size(x, 1)
        p = div(n, 2)

        for j = 1:p
            cx[j] = x[j] * (x[p+j] - 1.0) - 10 * x[p+j]
        end

        return 
    end

    function jac_c!(C, x)
        n = size(x, 1)
        N = div(n, 2)

        C .= 0
        for j = 1:N
            C[j, j] = x[N+j] - 1.0
            C[j, N+j] = x[j] - 10.0
        end

        return
    end

    # Sparse version of the jacobian 
    function jc!(C, x)
        n = size(x, 1)
        p = div(n, 2)

        for j = 1:p
            C[j, j] = x[p+j] - 1.0
            C[j, p+j] = x[j] - 10.0
        end
        return
    end

    # Sparsity pattern
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for j = 1:p 
        append!(jc_nzrows, [j, j])
        append!(jc_nzcols, [j, p+j]) 
    end

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x = 0.1 * ones(n)

    # return Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, xlow, xupp, x, n, m, p, Val(:only_equalities))

    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    xlow, xupp, x, n, m, p, Val(:only_equalities))

end