@testset "Slack variables handeling" begin
    
    # Case with one inequality constraint
    n_var = 3
    n_slack = 1
    p = 1
    p_eq = 0
    ineq_idx = p - n_slack + 1 : p

    g1(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
    c1(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2 - x[4]]
    al1(x) = dot(y,c1(x)) + mu/2 * dot(c1(x),c1(x))
    mu = 1.0

    y = [1.0]

    x0 = [-5, 5, 0.0]
    x = vcat(x0,g1(x0))
    alx0 = al1(x)
    gx0 = g1(x0)
    cx = c1(x)
    alx0 = al1(x)

    Traulls.step_slack!(x,y,cx,mu,n_slack,p)

    @test x[1:n_var] ≈ x0
    @test collect(ineq_idx) ≈ [1]
    @test size(ineq_idx,1) == n_slack
    @test x[n_var+1:end] ≈ (t -> max(0,t)).(1/mu * y[ineq_idx] .+ gx0)
    @test al1(x) <= alx0


    

    # Case with mixed equality and inequality constraints

    n_var = 3
    n_slack = 2
    p = 3
    p_eq = 1
    ineq_idx = p - n_slack + 1 : p

    g2(x) = [x[1]+x[2], 
    x[2]-x[3]]

    c2(x) = [x[1]^2 + x[2]^2 + x[3]^2,
    x[1]+x[2]-x[4], 
    x[2]-x[3]-x[5]]

    al2(x) = dot(y,c2(x)) + mu/2 * dot(c2(x),c2(x))

    mu = 1.0
    y = ones(p)
    x0 = collect(1:n_var)
    x = vcat(x0,g2(x0))
    gx0 = g2(x0)
    cx = c2(x)
    alx0 = al2(x)


    Traulls.step_slack!(x,y,cx,mu,n_slack,p)

    @test x[1:n_var] ≈ x0
    @test collect(ineq_idx) == [2,3]
    @test size(ineq_idx,1) == n_slack
    @test x[n_var+1:end] ≈ (t -> max(0,t)).(1/mu * y[ineq_idx] .+ gx0)
    @test al2(x) <= alx0

end
