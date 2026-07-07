# Problem 5.1 from Luksan & Vlsek collection (1999)
# Chained Rosenbrock with trigonometric-exponential constraints

function lv501(n::Int=100)
    @assert n >= 2 "n must be greater or equal than 2"

    m = 2(n-1)
    p = n-2

    # Residuals

    function r!(rx, x)
        n = size(x,1)
        N = n - 1

        for i = 1:N
            rx[i] = 10(x[i]^2 - x[i+1])
            rx[N+i] = x[i] - 1
        end

        return
    end

    function jac_r!(J, x)
        n = size(x, 1)
        N = n - 1

        J .= 0

        for i=1:n-1
            J[i, i] = 20x[i]
            J[i, i+1] = -10
            J[N+i, i] = -1
        end
        return
    end



    # Constraints

    function c!(cx, x)
        n = size(x,1)
        cx .= [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return
    end

    function jac_c!(A, x)
        n = size(x,1)
        A .= 0
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return
    end

    # Sparse version of jacobian and sparsity pattern
      function jr!(J, x)
        n = size(x,1)
        N = n - 1

        for i=1:n-1
            J[i, i] = 20x[i]
            J[i, i+1] = -10
            J[N+i, i] = -1
        end

        return
    end


    # Sparsity pattern residuals jacobian
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i = 1:n-1
        push!(jr_nzrows, i); push!(jr_nzcols, i)
        push!(jr_nzrows, i); push!(jr_nzcols, i+1)
        push!(jr_nzrows, n-1+i); push!(jr_nzcols, i)
    end

    function jc!(A, x)
        n = size(x,1)
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return
    end

    # Sparsity pattern constraints jacobian
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for k=1:n-2
        push!(jc_nzrows, k); push!(jc_nzcols, k)
        push!(jc_nzrows, k); push!(jc_nzcols, k+1)
        push!(jc_nzrows, k); push!(jc_nzcols, k+2)
    end

    xlow = fill(-Inf,n)
    xupp = fill(Inf,n)

    # Starting point
    x = [(mod(i,2) == 1 ? -1.2 : 1.0) for i=1:n]

    # return Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,xlow,xupp,x,n,m,p,Val(:only_equalities))
    return Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
    xlow, xupp, x, n, m, p, Val(:only_equalities))
end
