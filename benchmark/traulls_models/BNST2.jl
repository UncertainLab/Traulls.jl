# Example 2 from Biegler et al. (2000)
# Numerical Experience with a Reduced Hessian Method for Large Scale Constrained Optimization

function BNST2(n::Int=100)

    if n < 2
        @warn("BNST2: number of variables must be >= 2. Setting to 2")
        n = 2
    end

    m = n
    p = n - 1

    function r!(rx, x)
        rx .= x 
        return
    end

    function jac_r!(J, x)
       
        one_T = one(eltype(x))
        J .= 0
        for i in axes(x, 1)
            J[i, i] = one_T
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

        for j = 1:n-1
            cx[j] = x[1] * (x[j+1] - 1.0) - 10x[j+1]
        end

        return 
    end

    function jac_c!(C, x)
        n = size(x, 1)
        T = eltype(x)

        C .= 0.0
        for j = 1:n-1
            C[j, 1] = x[j+1] - 1.0
            C[j, j+1] = x[1] - 10.0
        end

        return
    end

    # Sparse version of the jacobian
    function jc!(C, x)
        n = size(x, 1)
        
        for j = 1:n-1
            C[j, 1] = x[j+1] - 1.0
            C[j, j+1] = x[1] - 10.0
        end
        return
    end

    # Sparsity pattern
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for j = 1:p
        append!(jc_nzrows, [j, j])
        append!(jc_nzcols, [1, j+1])
    end

    xlow = fill(-Inf, n)
    xupp = fill(Inf, n)

    x = 0.1 * ones(n)

    # return Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, xlow, xupp, x0, n, m, p, Val(:only_equalities))
    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    xlow, xupp, x, n, m, p, Val(:only_equalities))
end