@testset "Model with only inequality constraints" begin
    
    # Problem 65 form Hock-Schittkowski
    n = 3
    m = 3
    p = 1

    # Residuals 
    r(x) = [x[1]-x[2],
    (x[1]+x[2]-10)/3,
    x[3]-5.0]

    jac_r(x::Vector) = [1. -1. 0;
        1/3 1/3 0.;
        0. 0. 1.;]

    # Equality constraints
    c(x) = [48.0 - x[1]^2 - x[2]^2 - x[3]^2]
    jac_c(x) = [-2x[1] -2x[2] -2x[3]]

    # Bounds 
    x_low = [-4.5, -4.5, -5.0]
    x_upp = [4.5, 4.5, 5.0]

    x0 = [-5, 5, 0.0]
    x = vcat(x0,c(x0))

    model = Traulls.BoxCnls(r,c,jac_r,jac_c,x_low,x_upp,x0,n,m,p,false)

    @test size(x,1) == model.n
    @test Traulls.residuals(model,x) ≈ r(x0)
    @test model.n_slack == p 
    @test Traulls.jac_residuals(model,x) ≈ [1. -1. 0 0;
        1/3 1/3 0. 0;
        0. 0. 1. 0;]
    @test Traulls.jac_nlconstraints(model,x) ≈ hcat(jac_c(x0),[-1])
    @test Traulls.nlconstraints(model,x) ≈ zeros(1)
    @test model.nleq === nothing && model.jac_nleq === nothing

    
end
