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
    @test Traulls.vthv(H,v) ≈ dot(v,Hv_test)
end

@testset "SR1 Hessian structure test" begin

    n = 5   # parameters 
    m = 10  # residuals
    p = 3   # nonlinear constraints

    J = rand(m,n)
    C = rand(p,n)
    mu = 10.0
    v = rand(n)

    H = Traulls.HybridSR1(J,C,mu)
    H_test = J'*J + mu*C'*C

    
    @test H*v ≈  H_test*v
    @test H.small_res

    # Update test 
    y = 0.5*ones(p)     # Lagrange multipliers
    s = ones(n)         # step 
    g = 0.8*ones(n)     # gradient
    rx = rand(m)        # residuals
    cx = rand(p)        # nonlinear constraints 

    # value of the objective beforer and after taking step
    mx = 10. 
    mx_next = 8. 

    # update jacobians 
    J_next = J .+ 1
    C_next = C .+ 1

    H.secant_rhs = g - J'*rx - C'*(y+mu*cx)
    J .+= 1
    C .+= 1

    @test H.J ≈ J_next && H.C ≈ C_next

   Traulls.update_hessian!(H,s,mx,mx_next,1e-8,0.1)

    @test !H.small_res
    @test any(!≈(0.0), H.S)
    @test H.J ≈ J_next && H.C ≈ C_next

    # Computations with nonzero seocnd order terms 
    H.small_res = false 
    S = ones(n,n)
    H.S = S
    H_test = J_next'*J_next + mu*C_next'*C_next + S 
    
    @test H*v ≈  H_test*v

end
