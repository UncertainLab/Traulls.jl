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

    # TODO: test the updating procedures (one case where the update is done, one where it is rejected)
    # # Normal case update
    # y = ones(n)     # Lagrange multipliers
    # s = ones(n)      # step
    # denom_test = dot(y - H.S*s,s)

    # Traulls.update_hessian!(H, J, C, y, s)

    # @test H.J ≈ J && H.C ≈ C
    # @test abs(denom_test) > atol * (1 + norm(s) * norm(y-H.S*s)) && all(≠(0.0), H.S)

    # # Failed safeguard update
    # old_S = H.S
    # J .+= 1.0
    # C .+= 1.0

    # denom_test = dot(y-H.S*s,s)

    # Traulls.update_hessian!(H, J, C, y, s)
    # @test !(abs(denom_test) > max(atol, atol * norm(s) * norm(y-H.S*s)))
    # @test H.S ≈ old_S

    # v .+= 1.0
    # H_test = J'*J + mu*C'*C + old_S
    # @test H*v ≈ H_test*v

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

@testset "BFGS Hessian-vector product" begin

    n = 5   # parameters
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m, n)
    C = rand(p, n)
    mu = 10.0
    v = rand(n)

    H = Traulls.BFGS(J, C, mu)
    Hv = similar(v)

    # Second order terms initialized to the identity: H = JᵀJ + μCᵀC + I
    H_test = J'*J + mu*C'*C + I
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v

    # Matrix-vector product with a generic second order term
    S = rand(n, n)
    H.S .= S
    H_test = J'*J + mu*C'*C + S
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v
end

@testset "HybridBFGS Hessian-vector product" begin

    n = 5   # parameters
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m, n)
    C = rand(p, n)
    mu = 10.0
    v = rand(n)

    H = Traulls.HybridBFGS(J, C, mu)
    Hv = similar(v)

    # Second order terms initialized to the identity and small_res = false:
    # H = JᵀJ + μCᵀC + S
    @test H.small_res == false
    S = rand(n, n)
    H.S .= S
    H_test = J'*J + mu*C'*C + S
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v

    # Small residuals heuristic active: the second order term is dropped
    H.small_res = true
    H_test = J'*J + mu*C'*C
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v
end

@testset "HybridSR1 Hessian-vector product" begin

    n = 5   # parameters
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m, n)
    C = rand(p, n)
    mu = 10.0
    v = rand(n)

    H = Traulls.HybridSR1(J, C, mu)
    Hv = similar(v)

    # Second order terms initialized to zero and small_res = false:
    # H = JᵀJ + μCᵀC
    @test H.small_res == false
    H_test = J'*J + mu*C'*C
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v

    # Matrix-vector product with a generic second order term
    S = rand(n, n)
    H.S .= S
    H_test = J'*J + mu*C'*C + S
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v

    # Small residuals heuristic active: the second order term is dropped
    H.small_res = true
    H_test = J'*J + mu*C'*C
    mul!(Hv, H, v)
    @test Hv ≈ H_test*v
end
