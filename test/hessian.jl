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

    H = Traulls.SR1(J,C,mu)
    H_test = J'*J + mu*C'*C

    
    @test H*v ≈  H_test*v

    # Updated jacobians 
    J .+= 1.0
    C .+= 1.0

    @test H.J ≠ J 
    @test H.C ≠ C

    # Normal case update
    y = ones(n)     # Lagrange multipliers
    s = ones(n)      # step 
    denom_test = dot(y-H.S*s,s)

    Traulls.update_hessian!(H,J,C,y,s)

    @test H.J ≈ J && H.C ≈ C
    @test abs(denom_test) > atol * norm(s) * norm(y-H.S*s) && all(≠(0.0), H.S)
    
    # Failed safeguard update
    old_S = H.S
    J .+= 1.0
    C .+= 1.0

    denom_test = dot(y-H.S*s,s)

    Traulls.update_hessian!(H,J,C,y,s)
    @test !(abs(denom_test) > max(atol, atol * norm(s) * norm(y-H.S*s))) 
    @test H.S ≈ old_S

    v .+= 1.0
    H_test = J'*J + mu*C'*C + old_S
    @test H*v ≈ H_test*v


    
end
