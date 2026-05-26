@testset "Sparse model with nonlinear equality constraints" begin

    n = 100
    m = 2(n-1)
    p = n-2

    # Residuals

    function r!(rx, x)
        n = size(x,1)
        N = n - 1

        for i = 1:n-1
            rx[i] = 10(x[i]^2 - x[i+1])
            rx[N+i] = x[i] - 1
        end
        return
    end

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

    # Dense version
    function jr(x)
        n = size(x, 1)
        N = n-1
        m = 2N
        J = similar(x, m, n)

        J .= 0.0

        for i=1:n-1
            J[i, i] = 20x[i]
            J[i, i+1] = -10
            J[N+i, i] = -1
        end

        return J
    end
    # Constraints

    function c!(cx, x)
        n = size(x,1)
        cx .= [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return
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

    # Dense version
    function jc(x)
        n = size(x, 1)
        p = n-2

        A = similar(x, p, n)
        A .= 0.0

        for k=1:p
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end

        return A
    end

    xlow = fill(-Inf,n)
    xupp = fill(Inf,n)

    # Starting point
    x = [(mod(i,2) == 1 ? -1.2 : 1.0) for i=1:n]

    # Sparsity pattern residuals jacobian
    jr_nzrows = Vector{Int}([])
    jr_nzcols = Vector{Int}([])

    for i = 1:n-1
        push!(jr_nzrows, i); push!(jr_nzcols, i)
        push!(jr_nzrows, i); push!(jr_nzcols, i+1)
        push!(jr_nzrows, n-1+i); push!(jr_nzcols, i)
    end

    # Sparsity pattern constraints jacobian
    jc_nzrows = Vector{Int}([])
    jc_nzcols = Vector{Int}([])

    for k=1:n-2
        push!(jc_nzrows, k); push!(jc_nzcols, k)
        push!(jc_nzrows, k); push!(jc_nzcols, k+1)
        push!(jc_nzrows, k); push!(jc_nzcols, k+2)
    end

    model = Traulls.SparseCnlsModel!(r!, c!, jr!, jc!, jr_nzrows, jr_nzcols, jc_nzrows, jc_nzcols,
                             xlow, xupp, x, n, m, p, Val(:only_equalities))

    rx, cx = zeros(m), zeros(p)
    r!(rx, x); c!(cx, x)
    @test model.nlineq! === nothing && model.jnlineq! === nothing
    @test model.ncons == p && model.n == n
    @test Traulls.residuals(model, x) ≈ rx
    @test Traulls.nlconstraints(model, x) ≈ cx

    J = Traulls.jac_residuals(model, x)
    @test typeof(J) <: AbstractSparseMatrix
    (irows, icols, _) = findnz(J)
    @test size(irows, 1) == size(jr_nzrows, 1)
    @test size(icols, 1) == size(jr_nzcols, 1)
    @test Matrix(J) ≈ jr(x)

    C = Traulls.jac_nlconstraints(model, x)
    @test typeof(C) <: AbstractSparseMatrix
    (irows, icols, _) = findnz(C)
    @test size(irows, 1) == size(jc_nzrows, 1)
    @test size(icols, 1) == size(jc_nzcols, 1)
    @test Matrix(C) ≈ jc(x)


end
