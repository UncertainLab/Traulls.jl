@testset "Model with nonlinear inequalities and linear equalities" begin

    # Problem 225 from Hock & Schittkowski collection

    # Dimensions 
    n = 3
    m = 2
    p = 4

    # Residuals
    r(x) = [x[1],x[2]]
    
    jac_r(x) = [1.0 0.0 0.0;
    0.0 1.0 0.0]

    # Nonlinear constraints
    c(x) = [x[1]^2 + x[2]^2 - 1,
    9x[1]^2 + x[2]^2 - 9,
    x[1]^2 - x[2],
    x[2]^2 - x[1]]

    jac_c(x) = [2x[1] 2x[2] 0.0;
    18x[1] 2x[2] 0.0;
    2x[1] -1.0 0.0;
    -1.0 2x[2] 0.0]

    # Linear constraints 
    A = [1.0 1.0 -1.0]
    b = [1.0]

    x_low = [-Inf,-Inf,0.0]
    x_upp = [Inf, Inf, Inf]

    x0 = [3.0, 1., 3.]

    model = Traulls.PolyhedralCnls(r,c,jac_r,jac_c,A,b,x_low,x_upp,x0,n,m,p,false)    
end