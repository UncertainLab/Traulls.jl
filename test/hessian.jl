@testset "Gauss-Newton Hessian structure test" begin
    
    n = 5   # parameters 
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m,n)
    C = rand(p,n)
    mu = rand()
    v = rand(n)

    H = Traulls.GN(J,C,mu)
    H_test = J'*J + mu*C'*C

    Hv_test = H_test*v
    @test H*v ≈ Hv_test

    # Update 
    J .+= 1.0
    C .+= 1.0


    @test H.J ≠ J 
    @test H.C ≠ C
    Traulls.update_hessian!(H,J,C)
    H_test = J'*J + mu*C'*C

    Hv_test = H_test*v
    @test H*v ≈ Hv_test

    # Reset procedure 
    J .= 10
    C .= 10
    new_mu = 10.0

    Traulls.reset_hessian!(H,J,C,new_mu)
    @test H.J ≈ J && H.C ≈ C
    @test H.mu ≠ mu
end

@testset "SR1 Hessian structure test" begin

    atol = sqrt(eps(Float64)) # tolerance
    n = 5   # parameters 
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m,n)
    C = rand(p,n)
    mu = 10.0
    v = rand(n)

    H = Traulls.SR1(J, C, mu)
    H_test = J'*J + mu*C'*C

    
    @test H*v ≈  H_test*v

    # Updated jacobians 
    J .+= 1.0
    C .+= 1.0
    S = 2*ones(n,n)
    H.S = copy(S)
    @test H.J ≠ J 
    @test H.C ≠ C
    H.J .= J
    H.C .= C
    @test all(!iszero, H.S)

    H_test = J'*J + mu*C'*C + S
    @test H*v ≈ H_test*v

    # Reset procedure 
    J .= 10
    C .= 10
    new_mu = 100.0
    Traulls.reset_hessian!(H, J, C, new_mu)
    @test H.J ≈ J && H.C ≈ C
    @test H.S ≈ zeros(n,n)
    @test H.mu ≠ mu && H.mu ≈ new_mu

    H_test = J'*J + new_mu*C'*C
    @test H*v ≈ H_test*v
end

@testset "Secant Hessian approximations" begin

    n = 5   # parameters
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m, n)
    C = rand(p, n)
    mu = 10.0
    v = rand(n)

    # Data of an accepted step: J₊ is chosen so that the secant right-hand side
    # ŷ = (J₊ - J)ᵀr₊ + (C₊ - C)ᵀ(y + μc₊) equals a prescribed vector with positive curvature
    s = rand(n)
    M = rand(n, n)
    rhs = (M'*M + I) * s
    rx = rand(m)
    cx = rand(p)
    y = rand(p)
    J_new = J + rx * rhs' / dot(rx, rx)
    g = J_new' * rx + C' * (y + mu * cx)

    first_order(H, v) = (H.J' * H.J + mu * H.C' * H.C) * v
    second_order(H, v) = H * v - first_order(H, v)

    @testset "$name" for (name, H) in (("SR1", Traulls.SR1(J, C, mu)),
                                        ("BFGS", Traulls.BFGS(J, C, mu)),
                                        ("L-SR1", Traulls.LSR1(J, C, mu)))
        # Initial second order terms: 0 for SR1, identity for BFGS and L-SR1
        S0 = H isa Traulls.SR1 ? zeros(n, n) : Matrix(1.0I, n, n)
        @test H * v ≈ first_order(H, v) + S0 * v

        # The updated approximation satisfies the secant equation and holds the new Jacobians
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.5, 1.0, g, y, s)
        @test H.J ≈ J_new && H.C ≈ C
        @test second_order(H, s) ≈ rhs

        # Reset procedure
        Traulls.reset_hessian!(H, J, C, 2mu)
        @test H.J ≈ J && H.mu ≈ 2mu
        @test H * v ≈ (J' * J + 2mu * C' * C + S0) * v
    end

    @testset "Skipped updates" begin
        # SR1: the update is skipped when the secant equation already holds
        H = Traulls.SR1(J, C, mu)
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.5, 1.0, g, y, s)
        S_old = copy(H.S)
        H.J .= J # same secant pair as for the first update
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.5, 1.0, g, y, s)
        @test H.S ≈ S_old && H.nupdates == 1

        # BFGS: the update is skipped for a step with negative curvature
        H = Traulls.BFGS(J, C, mu)
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.5, 1.0, g, y, -s)
        @test H.S ≈ I && H.nupdates == 0
    end

    @testset "Hybrid switching" begin
        H = Traulls.HybridSR1(J, C, mu)
        @test !H.small_res

        # Fast decrease of the objective: second order terms are updated but not used
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.1, 1.0, g, y, s)
        @test H.small_res
        @test H * v ≈ first_order(H.approx, v)
        @test second_order(H.approx, s) ≈ rhs

        # Slow decrease of the objective: second order terms are used
        H.approx.J .= J
        Traulls.update_hessian!(H, J_new, C, rx, cx, 0.95, 1.0, g, y, s)
        @test !H.small_res
        @test H * v ≈ H.approx * v

        Traulls.reset_hessian!(H, J, C, mu)
        @test !H.small_res && H.approx.S ≈ zeros(n, n)
    end
end

@testset "Solving with every Hessian approximation" begin

    # Large residual problem with a nonlinear equality constraint
    r!(rx, x) = (rx .= [x[1]^2 + x[2] - 11, x[1] + x[2]^2 - 7, x[1] - x[2] + 3, 10sin(x[1]*x[2])]; nothing)
    jac_r!(J, x) = (J .= [2x[1] 1; 1 2x[2]; 1 -1; 10x[2]*cos(x[1]*x[2]) 10x[1]*cos(x[1]*x[2])]; nothing)
    c!(cx, x) = (cx .= [x[1]^2 + x[2]^2 - 9]; nothing)
    jac_c!(C, x) = (C .= [2x[1] 2x[2]]; nothing)

    objectives = Dict{Symbol, Float64}()
    for hessian_approx in keys(Traulls.dict_hessians)
        model = Traulls.CnlsModel!(r!, c!, jac_r!, jac_c!, [-5.0, -5.0], [5.0, 5.0], [1.0, 2.0],
                                   2, 4, 1, Val(:only_equalities))
        results = traulls(model; hessian_approx = hessian_approx)
        @test results.status == Traulls.first_order_critical
        objectives[hessian_approx] = results.objective
    end
    @test all(f -> isapprox(f, objectives[:gn]; rtol = 1e-6), values(objectives))
end
