@testset "Model with only inequality constraints" begin
    # Problem 65 form Hock-Schittkowski
    n = 3
    m = 3
    p = 1

    # Residuals
    r(x) = [x[1]-x[2],
    (x[1]+x[2]-10)/3,
    x[3]-5.0]

    function r!(rx, x)
        rx .= [x[1]-x[2],
               (x[1]+x[2]-10)/3,
               x[3]-5.0]
        return
    end

    jac_r(x) = [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]

    function jac_r!(J, x)
        J .= [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]
        return
    end
    # Equality constraints
    c(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
    jac_c(x) = [-2x[1] -2x[2] -2x[3]]

    function c!(cx, x)
        cx .= [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
        return
    end

    function jac_c!(C, x)
        C .= [-2x[1] -2x[2] -2x[3]]
        return
    end

    # Bounds
    x_low = [-4.5, -4.5, -5.0]
    x_upp = [4.5, 4.5, 5.0]

    x0 = [-5, 5, 0.0]
    x = vcat(x0,c(x0))

    # Testing for model defined with in-place methods
    model = Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,
                             Val(:only_inequalities))

    @test size(x,1) == model.n
    @test Traulls.residuals(model,x) ≈ r(x0)
    @test model.nslack == p && model.ncons == p
    @test Traulls.jac_residuals(model,x) ≈ [1. -1. 0 0;
        1/3 1/3 0. 0;
        0. 0. 1. 0;]
    @test Traulls.jac_nlconstraints(model,x) ≈ hcat(jac_c(x0),[-1])
    @test Traulls.nlconstraints(model,x) ≈ zeros(1)
    @test model.nleq! === nothing && model.jac_nleq! === nothing
    @test model.nlincons == 0

    # Testing for model defined with standard functions
    model = Traulls.CnlsModel(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,Val(:only_inequalities))

    @test size(x,1) == model.n
    @test Traulls.residuals(model,x) ≈ r(x0)
    @test model.nslack == p
    @test Traulls.jac_residuals(model,x) ≈ [1. -1. 0 0;
        1/3 1/3 0. 0;
        0. 0. 1. 0;]
    @test Traulls.jac_nlconstraints(model,x) ≈ hcat(jac_c(x0),[-1])
    @test Traulls.nlconstraints(model,x) ≈ zeros(1)
    @test model.nleq! === nothing && model.jac_nleq! === nothing
end

@testset "Model with only equality constraints" begin
    n = 10
    m = 2(n-1)
    p = n-2

    # Residuals

    function r(x)
        n = length(x)
        m = 2(n-1)
        rx = Vector{eltype(x)}(undef,m)
        rx[1:n-1] = [10(x[i]^2 - x[i+1]) for i=1:n-1]
        rx[n:m] = [x[k-n+1] - 1 for k=n:m]
        return rx
    end

    function r!(rx, x)
        n = size(x,1)
        m = 2(n-1)

        rx[1:n-1] .= [10(x[i]^2 - x[i+1]) for i=1:n-1]
        rx[n:m] .= [x[k-n+1] - 1 for k=n:m]
        return
    end

    function jac_r(x)
        n = size(x,1)
        m = 2(n-1)
        J = zeros(eltype(x), (m,n))

        for i=1:n-1
            J[i,i] = 20x[i]
            J[i,i+1] = -10
        end

        for i=n:m
            J[i,i-n+1] = 1
        end
        return J
    end

    function jac_r!(J, x)
        n = size(x,1)
        m = 2(n-1)

        J .= 0.0
        for i=1:n-1
            J[i,i] = 20x[i]
            J[i,i+1] = -10
        end

        for i=n:m
            J[i,i-n+1] = 1
        end
        return
    end

    # Constraints

    function c(x)
        n = length(x)
        cx = [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return cx
    end

    function c!(cx, x)
        n = length(x)
        cx .= [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+1] -
            x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
        return cx
    end

    function jac_c(x)
        n = size(x,1)
        A = zeros(eltype(x), (n-2,n))
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return A
    end

    function jac_c!(A, x)
        n = size(x,1)
        A .= 0.0
        for k=1:n-2
            A[k,k] = -(x[k]+1) * exp(x[k]-x[k+1])
            A[k,k+1] = 9x[k+1]^2 + cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2]) + 4 + x[k]*exp(x[k]-x[k+1])
            A[k,k+2] = 2 - cos(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + sin(x[k+1]-x[k+2])*cos(x[k+1]+x[k+2])
        end
        return
    end

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)


    # Starting point
    x0 = [(mod(i,2) == 1 ? -1.2 : 1.0) for i=1:n]

    model = Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,
                             Val(:only_equalities))

    @test size(x0,1) == model.n
    @test model.nslack == 0
    @test model.ncons == p
    @test Traulls.residuals(model,x0) ≈ r(x0)
    @test Traulls.jac_residuals(model,x0) ≈ jac_r(x0)
    @test Traulls.jac_nlconstraints(model,x0) ≈ jac_c(x0)
    @test Traulls.nlconstraints(model,x0) ≈ c(x0)
    @test model.nlineq! === nothing && model.jac_nlineq! === nothing

    # Testing for model defined with standard functions
    model = Traulls.CnlsModel(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,Val(:only_equalities))

    @test size(x0,1) == model.n
    @test model.nslack == 0
    @test model.ncons == p
    @test Traulls.residuals(model,x0) ≈ r(x0)
    @test Traulls.jac_residuals(model,x0) ≈ jac_r(x0)
    @test Traulls.jac_nlconstraints(model,x0) ≈ jac_c(x0)
    @test Traulls.nlconstraints(model,x0) ≈ c(x0)
    @test model.nlineq! === nothing && model.jac_nlineq! === nothing
end

@testset "Model with a mix of equalities and inequalities" begin

    n = 5
    m = 4
    p_eq = 1
    p_ineq = 1

    # Residuals

    r(x) = [exp((x[i]-i)^2 / 2x[i+1]^2) for i=1:n-1]

    function r!(rx, x)
        rx .= [exp((x[i]-i)^2 / 2x[i+1]^2) - i for i=1:n-1]
    end

    function jac_r(x)
        n = size(x,1)
        J = zeros(n-1,n)

        for i=1:n-1
            J[i,i] = 2x[i] * exp((x[i]-i)^2 / 2x[i+1]^2)
            J[i,i+1] = exp((x[i]-i)^2 / 2x[i+1]^2) / x[i+1]^3
        end

        return J
    end

    function jac_r!(J, x)
        J .= 0
        for i=1:size(x,1)-1
            J[i,i] = 2x[i] * exp((x[i]-i)^2 / 2x[i+1]^2)
            J[i,i+1] = exp((x[i]-i)^2 / 2x[i+1]^2) / x[i+1]^3
        end
        return
    end

    # Constraints

    g(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

    function g!(gx, x)
        gx .= [1 - x[1]^2 - x[2]^2 - x[3]^2]
        return
    end

    jac_g(x) =  [-2x[1] -2x[2] -2x[3] 0.0 0.0]

    function jac_g!(Gx, x)
        Gx .= [-2x[1] -2x[2] -2x[3] 0.0 0.0]
    end

    h(x) = [x[4]^2 + x[5]^2 - 1]

    function h!(hx, x)
        hx .= [x[4]^2 + x[5]^2 - 1]
        return
    end

    jac_h(x) = [0.0 0.0 0.0 2x[4] 2x[5]]

    function jac_h!(Hx, x)
        Hx .= [0.0 0.0 0.0 2x[4] 2x[5]]
        return
    end

    x_low = [0.1^i for i=1:n]
    x_upp = [Float64(i^2) for i=1:n]

    # Starting point
    x0 = [1/i for i=1:n]

    # Model with in place methods
    model = Traulls.CnlsModel!(r!,h!,g!,jac_r!,jac_h!,jac_g!,x_low,x_upp,x0,n,m,
                             p_eq,p_ineq)

    @test n+p_ineq == model.n
    @test model.nslack == 1
    @test model.ncons == p_eq+p_ineq
    @test size(model.x,1) == model.n
    @test Traulls.residuals(model,model.x) ≈ r(x0)
    @test Traulls.jac_residuals(model,model.x) ≈ hcat(jac_r(x0),zeros(m))
    @test Traulls.nlconstraints(model,model.x) ≈ vcat(h(x0),g(x0)-g(x0))
    @test Traulls.jac_nlconstraints(model,model.x) ≈ vcat(hcat(jac_h(x0),zeros(1,p_ineq)),
                                                  hcat(jac_g(x0), Matrix{Float64}(-I,p_ineq,p_ineq)))

    # Model with out of place methods
    model = Traulls.CnlsModel(r,h,g,jac_r,jac_h,jac_g,x_low,x_upp,x0,n,m,p_eq,p_ineq)

    @test n+p_ineq == model.n
    @test model.nslack == 1
    @test model.ncons == p_eq+p_ineq
    @test size(model.x,1) == model.n
    @test Traulls.residuals(model,model.x) ≈ r(x0)
    @test Traulls.jac_residuals(model,model.x) ≈ hcat(jac_r(x0),zeros(m))
    @test Traulls.nlconstraints(model,model.x) ≈ vcat(h(x0),g(x0)-g(x0))
    @test Traulls.jac_nlconstraints(model,model.x) ≈ vcat(hcat(jac_h(x0),zeros(1,p_ineq)),
                                                  hcat(jac_g(x0), Matrix{Float64}(-I,p_ineq,p_ineq)))


end

@testset "Model with autodiff" begin

    n = 9
    N = div(n-1,4)
    m = 4N
    p = 3*N


    # Residuals
    function r!(rx,x)
        N = div(size(x,1)-1,4)

        rx[1:N] .= [(x[4i-3] - x[4i-2])^2 for i=1:N]
        rx[N+1:2N] .= [x[4i-2] + x[4i-1] - 2 for i=1:N]
        rx[2N+1:3N] .= [x[4i] - 1 for i=1:N]
        rx[3N+1:4N] .= [x[4i+1] - 1 for i=1:N]

        return
    end

    function r(x)
        m = 4*div(size(x,1)-1,4)
        res = similar(x,m)
        r!(res,x)
        return res
    end
    # Inversed arguments for ForwardDiff.jacobian! call
    jac_r!(J, x) = ForwardDiff.jacobian!(J, r!, zeros(m), x)

    jac_r(x) = ForwardDiff.jacobian(r,x)
    # Constraints

    function c!(cx, x)
        N = div(size(x,1)-1,4)

        for k = 1:3N
            l = 4*div(k-1,3)
            if mod(k,3) == 1
                cx[k] = x[l+1]^2 + 3*x[l+2]
            elseif k % 3 == 2
                cx[k] = x[l+3]^2 + x[l+4] - 2*x[l+5]
            else
                cx[k] = x[l+2]^2 - x[l+5]
            end
        end
        return
    end

    function c(x)
        p = 3*div(size(x,1)-1,4)
        res = similar(x,p)
        c!(res, x)
        return res
    end

    jac_c(x) = ForwardDiff.jacobian(c,x)

    jac_c!(C, x) = ForwardDiff.jacobian!(C, c!, zeros(3N), x)

    x_low = fill(-Inf,n)
    x_upp = fill(Inf,n)

    # Starting point

    x0 = 2 .* ones(n)

    model = Traulls.CnlsModel!(r!,c!,jac_r!,jac_c!,x_low,x_upp,x0,n,m,p,Val(:only_equalities))

    @test Traulls.jac_residuals(model,x0) ≈ jac_r(x0)
    @test Traulls.jac_nlconstraints(model,x0) ≈ jac_c(x0)

end
